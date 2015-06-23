#include <thread>
#include <vector>
#include <map>
#include <glog/logging.h>
#include "trainer/trainer.h"
#include "mshadow/tensor.h"
using std::vector;
using std::map;

namespace singa {
int ProcsIDOf(int group_id, int id, int flag){
  int procsid=-1;
  auto cluster=Cluster::Get();
  if(flag==kServer){
    procsid=group_id*cluster->nservers_per_group()/
      cluster->nservers_per_procs()+id/cluster->nservers_per_procs();
    if(cluster->server_worker_separate())
      procsid+=cluster->nworker_procs();
  }else if(flag==kWorkerLayer || flag==kWorkerParam){
    procsid=group_id*cluster->nworkers_per_group()
      /cluster->nworkers_per_procs();
    if(cluster->nworkers_per_group()>cluster->nworkers_per_procs())
      procsid+=id/cluster->nworkers_per_procs();
  }else{
    LOG(ERROR)<<"Unkown flag ("<<flag<<")";
  }
  return procsid;
}

void Trainer::RegisterDefaultClasses(const singa::ModelProto& proto){
  // register all layers appearing in the neural net
  singa::NeuralNet::RegisterLayers();
  Singleton<Factory<singa::Param>>::Instance()->Register(
      "Param", CreateInstance(singa::Param, singa::Param));
  Singleton<Factory<singa::Updater>>::Instance() ->Register(
      "Updater", CreateInstance(singa::SGDUpdater, singa::Updater));
}

typedef struct HandleContext_{
  shared_ptr<Dealer> dealer;
  int group_id, id;
} HandleContext;

void HandleWorkerFinish(void * ctx){
  HandleContext* hctx=static_cast<HandleContext*> (ctx);
  Msg* msg=new Msg();
  msg->set_src(-1,-1, kRuntime);
  msg->set_dst(hctx->group_id, hctx->id, kServer);
  msg->set_type(kStop);
  hctx->dealer->Send(&msg);
}

void Trainer::Start(const ModelProto& mproto, const ClusterProto& cproto,
    int procs_id){
  procs_id_=procs_id;
  RegisterDefaultClasses(mproto);

  auto cluster=Cluster::Get(cproto, procs_id);
  router_=make_shared<Router>();
  router_->Bind(kInprocRouterEndpoint);
  if(cluster->nprocs()>1)
    router_->Bind(cluster->endpoint());

  // create servers
  vector<shared_ptr<Server>> servers;
  vector<HandleContext> ctx;
  int nthreads=1; // the first socket is the router
  if(cluster->has_server()){ // todo move sever creation to a method
    int pid=cluster->procs_id();
    if(cluster->server_worker_separate())
      pid-=cluster->nworker_procs();
    int gid=pid*cluster->nservers_per_procs()/cluster->nservers_per_group();
    int start=pid*cluster->nservers_per_procs()%cluster->nservers_per_group();
    int end=start+cluster->nservers_per_group();
    // the ParamShard for servers consists of a dictionary of Param objects
    auto shard=make_shared<Server::ParamShard>();
    if(start<end){
      auto dealer=make_shared<Dealer>();
      dealer->Connect(kInprocRouterEndpoint);
      for(int sid=start;sid<end;sid++){
        auto server=make_shared<Server>(nthreads++, gid, sid);
        server->Setup(mproto.updater(), shard);
        servers.push_back(server);
        HandleContext hc{dealer, gid, sid};
        ctx.push_back(hc);
        CHECK(cluster->runtime()->sWatchSGroup(gid, sid, HandleWorkerFinish,
            &ctx.back()));
      }
    }
  }
  // create workers
  vector<shared_ptr<Worker>> workers;
  std::map<int, shared_ptr<Trainer::ParamShard>> shards;
  if(cluster->has_worker()){ //move worker creation to a method
    auto net=NeuralNet::SetupNeuralNet(mproto.neuralnet(), kTrain,
        cluster->nworkers_per_group());
    //LOG(ERROR)<<net->ToString();
    int pid=cluster->procs_id();
    int gstart, gend, wstart, wend;
    if(cluster->nworkers_per_group()>=cluster->nworkers_per_procs()){
      // all workers in this procs are from the same group
      gstart=pid*cluster->nworkers_per_procs()/cluster->nworkers_per_group();
      gend=gstart+1;
      wstart=pid*cluster->nworkers_per_procs()%cluster->nworkers_per_group();
      wend=wstart+cluster->nworkers_per_group();
    }else{
      // there are multiple groups in this procs
      CHECK_EQ(cluster->nworkers_per_procs()%cluster->nworkers_per_group(),0);
      int groups_per_procs=
        cluster->nworkers_per_procs()/cluster->nworkers_per_group();
      gstart=pid*groups_per_procs;
      gend=(pid+1)*groups_per_procs;
      wstart=0;
      wend=cluster->nworkers_per_group();
    }
    for(int gid=gstart;gid<gend;gid++){
      shared_ptr<NeuralNet> train_net, test_net, validation_net;
      if(gid==gstart)
        train_net=net;
      else{
        train_net=NeuralNet::SetupNeuralNet(mproto.neuralnet(), kTrain,
            cluster->nworkers_per_group());
        // the train net for other groups may share parameter values from the
        // first group
        if(cluster->share_memory())
          train_net->ShareParams(net, kValueOnly);
      }
      if(gid==0){
        // validation and test are performed only by the first group
        if(mproto.test_steps()){
          test_net=NeuralNet::SetupNeuralNet(mproto.neuralnet(), kTest,
              cluster->nworkers_per_group());
          if(test_net!=nullptr)
            test_net->ShareParams(train_net, kValueOnly);
        }
        if(mproto.validation_steps()){
          validation_net=NeuralNet::SetupNeuralNet(mproto.neuralnet(), kValidation,
              cluster->nworkers_per_group());
          if(validation_net!=nullptr)
            validation_net->ShareParams(train_net, kValueOnly);
        }
      }
      // create ParamShard for the workers
      auto shard=make_shared<Trainer::ParamShard>();
      shards[gid]=shard;
      for(auto layer: train_net->layers()){
        int procsid=ProcsIDOf(gid, layer->partitionid(),kWorkerParam);
        bool local=procsid==cluster->procs_id();
        for(auto param: layer->GetParams()){
          int owner_procs=param->owner()==param->id()?procsid:procs_id_;
          if(shard->find(param->owner())==shard->end())
            (*shard)[param->owner()]=
              make_shared<ParamInfo>(param, local, owner_procs);
          else
            shard->at(param->owner())->AddParam(param, local);
        }
      }
      for(int wid=wstart;wid<wend;wid++){
        shared_ptr<Worker> worker=nullptr;
        if(mproto.alg()==ModelProto_GradCalcAlg_kBackPropagation)
          worker=make_shared<BPWorker>(nthreads++,gid, wid);
        else{
          worker=make_shared<CDWorker>(nthreads++,gid, wid);
        }
        worker->Setup(mproto, train_net);
        worker->set_test_net(test_net);
        worker->set_validation_net(validation_net);
        workers.push_back(worker);
      }
    }
  }

#ifdef USE_MPI
  for(int i=0;i<nSocket;i++){
    MPIQueues.push_back(make_shared<SafeQueue>());
  }
#endif
  vector<std::thread> threads;
  for(auto server: servers)
    threads.push_back(std::thread(&Server::Run,server.get()));
  for(auto worker: workers)
    threads.push_back(std::thread(&Worker::Run,worker.get()));
  Run(workers.size(), servers.size(), shards);
  for(auto& thread: threads)
    thread.join();
}

void Trainer::Run(int nworkers, int nservers,
    const std::map<int, shared_ptr<Trainer::ParamShard>>& shards){
  auto cluster=Cluster::Get();
  procs_id_=cluster->procs_id();
  map<int, shared_ptr<Dealer>> interprocs_dealers;
  Metric perf;
  bool stop=false;
  while(!stop){
    Msg* msg=router_->Receive();
    if(msg==nullptr){
      LOG(ERROR)<<"Connection broken!";
      exit(0);
    }
    while(msg!=nullptr){
      int dst_flag=msg->dst_flag();
      int type=msg->type();
      int dst_procs=msg->dst_first();
      if(dst_flag == kStub&&(dst_procs==procs_id_||dst_procs==-1)){
        if(type==kConnect){
          msg =HandleConnect(&msg);
        }else if(type==kStop){
          if(msg->src_flag()==kServer)
            nservers--;
          else if (msg->src_flag()==kWorkerParam)
            nworkers--;
          delete msg;
          msg=nullptr;
          if(nworkers==0&&nservers==0){
            stop=true;
            break;
          }
        }else if(type==kMetric){
          if(msg->src_first()==0){
            int step=msg->target_first();
            string prefix((char*)msg->frame_data(), msg->frame_size());
            msg->next_frame();
            Metric cur;
            cur.ParseString(string((char*)msg->frame_data(), msg->frame_size()));
            perf.AddMetrics(cur);
            LOG(ERROR)<<prefix<<" step-" <<step<<", "<<perf.ToString();
            perf.Reset();
          }
          DeleteMsg(&msg);
        }else if(cluster->nserver_groups()>0){
          int group_id=msg->src_first();
          int paramid=msg->target_first();
          auto entry=shards.at(group_id)->at(paramid);
          switch (type){ // TODO process other requests, e.g. RESTful
            case kUpdate:
              msg=HandleUpdate(entry, &msg);
              break;
            case kRUpdate:
              HandleUpdateResponse(entry, &msg);
              break;
            case kGet:
              msg=HandleGet(entry, &msg);
              break;
            case kRGet:
              msg=HandleGetResponse(entry, &msg);
              break;
            case kPut:
              msg=HandlePut(entry, &msg);
              break;
            default:
              break;
          }
        }else{
          delete msg;
          msg=nullptr;
        }
      }else{
        int dst_procs_id;
        if(dst_flag==kStub){
          dst_procs_id=msg->dst_first();
        }else{
          dst_procs_id=ProcsIDOf(msg->dst_first(), msg->dst_second(), msg->dst_flag());
        }
        if(dst_procs_id!=procs_id_){
        /*
          // forward to other procs
          if (interprocs_dealers.find(procs_id)==interprocs_dealers.end())
          interprocs_dealers[procs_id]=make_shared<Dealer>(procs_id);
          interprocs_dealers[procs_id]->Send(&msg);
          */
        }else{
          router_->Send(&msg);
        }
      }
    }
  }
  /*
  perf.Avg();
  if(perf_step>=0)
    LOG(ERROR)<<perf_prefix<<" step-"<<perf_step<<", "<<perf.ToString();
    */
}
Msg* Trainer::HandleConnect(Msg** msg){
  string ping((char*)(*msg)->frame_data(), (*msg)->frame_size());
  CHECK_STREQ("PING", ping.c_str());
  // ping-pong for debug
  (*msg)->SwapAddr();
  Msg* reply=new Msg();
  reply->SetAddr(*msg);
  reply->add_frame("PONG", 4);
  reply->set_type(kConnect);
  delete *msg;
  *msg=NULL;
  return reply;
}
int Trainer::Sharding(int param_id){
  return param_id%Cluster::Get()->nservers_per_group();
}
/*
int Worker::Sharding(int param_id){
  static map<int, int> id2procs;
  if(id2procs.find(param_id)==id2procs.end()){
  auto cluster=Cluster::Get();
  int server_group=group_id_%cluster->nserver_groups();
  int nprocs_per_server_group=
    cluster->nservers_per_group()/cluster->nservers_per_procs();
  int procsid=server_group*nprocs_per_server_group+
    param_id%nprocs_per_server_group;
  procsid= cluster->server_worker_separate()?
    cluster->nworker_procs()+procsid:procsid;
  id2procs[param_id]=procsid;
  }
  return id2procs[param_id];
}
*/


Msg* Trainer::HandleGet(shared_ptr<ParamInfo> pi, Msg** msg){
  Msg* msgg=*msg, *reply=nullptr;
  int version=msgg->target_second();
  if(msgg->src_flag()==kStub){
    if(version<=pi->shares.at(0)->version()){
      reply=pi->shares.at(0)->HandleGetMsg(msg);
    }else if(version>pi->next_version){
      // reinsert into a msg queue.
    }
  }else if(version>pi->next_version){
    pi->next_version=version;
    int gid=msgg->src_first(), pid=msgg->target_first();
    int dstgroup=gid/Cluster::Get()->nworker_groups_per_server_group();
    int dstid=Sharding(pid);
    int dstprocs=ProcsIDOf(dstgroup, dstid, kServer);
    reply=pi->shares.at(0)->GenGetMsg(dstprocs!=procs_id_);
    reply->set_src(procs_id_, gid, kStub);
    reply->set_dst(dstgroup, dstid, kServer);
  }
  return reply;
}

Msg* Trainer::HandleGetResponse(shared_ptr<ParamInfo>pi, Msg** msg){
  pi->shares.at(0)->ParseGetResponseMsg(msg);
  return nullptr;
  // process get requests in waiting queue
}

Msg* Trainer::HandleUpdate(shared_ptr<ParamInfo>pi, Msg** msg){
  Msg* msgg=*msg, *update=nullptr;
  int step= msgg->target_second();
  if(msgg->src_flag()==kStub){
    if(pi->num_update<pi->num_local)
      return *msg; //wait unitl local updates are ready
    int n;
    sscanf((char*)(*msg)->frame_data(), "%d", &n);
    pi->num_update+=n;
    auto it=pi->shares.begin();
    auto shape=mshadow::Shape1((*it)->size());
    mshadow::Tensor<mshadow::cpu,1> agg((*it)->mutable_cpu_grad(), shape);
    mshadow::Tensor<mshadow::cpu,1> grad((*it)->mutable_cpu_grad(), shape);
    agg+=grad;
  }else if(++pi->num_update>=pi->num_local){
    auto it=pi->shares.begin();
    auto shape=mshadow::Shape1((*it)->size());
    mshadow::Tensor<mshadow::cpu,1> agg((*it)->mutable_cpu_grad(), shape);
    for(++it;it!=pi->shares.end();it++){
      mshadow::Tensor<mshadow::cpu,1> grad((*it)->mutable_cpu_grad(), shape);
      agg+=grad;
    }
    agg/=pi->num_total;
    if(pi->num_local<pi->num_total){
      update=pi->shares.at(0)->GenUpdateMsg(pi->owner_procs!=procs_id_, step);
      int gid=msgg->src_first();
      update->set_src(procs_id_, gid,kStub);
      update->set_dst(pi->owner_procs, gid, kStub);
      pi->num_update=0;
    }
  }
  if(pi->num_update==pi->num_total){
    int gid=msgg->src_first();
    int dstgroup=gid/Cluster::Get()->nworker_groups_per_server_group();
    int dstid=Sharding(msgg->target_first());
    int dstprocs=ProcsIDOf(dstgroup, dstid, kServer);
    update=pi->shares.at(0)->GenUpdateMsg(dstprocs!=procs_id_, step);
    update->set_src(procs_id_, gid, kStub);
    update->set_dst(dstgroup, dstid, kServer);
    pi->num_update=0;
  }
  delete *msg;
  *msg=NULL;
  return update;
}

int Trainer::HandleUpdateResponse(shared_ptr<Trainer::ParamInfo> pi, Msg** msg){
  HandleGetResponse(pi, msg);
  return 1;
}

Msg* Trainer::HandlePut(shared_ptr<Trainer::ParamInfo>pi, Msg** msg){
  CHECK_NE((*msg)->src_flag(), kStub);
  int gid=(*msg)->src_first();
  int id=(*msg)->target_first();
  int dstgroup=gid/Cluster::Get()->nworker_groups_per_server_group();
  int dstid=Sharding(id);
  int dstprocs=ProcsIDOf(dstgroup, dstid, kServer);
  Msg* put=pi->shares.at(0)->GenPutMsg(dstprocs!=procs_id_);
  put->set_src(procs_id_, gid , kStub);
  put->set_dst(dstgroup, dstid, kServer);
  delete *msg;
  *msg=NULL;
  return put;
}
} /* singa */
