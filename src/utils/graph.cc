
#include "utils/graph.h"
#include <glog/logging.h>
#include <algorithm>
#include <queue>
#include <unordered_set>

namespace singa {
/************************Node********************************/

Node::~Node() {
  // the proto field is deleted outside by other functions
}

Node::Node(string name) {
  this->name = name;
}

Node::Node(const string& name, const string& origin, int id, void* proto) {
  this->name = name;
  this->origin = origin;
  this->proto = proto;
  this->partition_id = id;
}

void Node::AddDstNode(Node* dstnode) {
  dstnodes.push_back(dstnode);
}

void Node::AddSrcNode(Node* srcnode) {
  srcnodes.push_back(srcnode);
}

void Node::RemoveDstNode(Node* dst) {
  auto iter = dstnodes.begin();
  while ((*iter)->name != dst->name && iter != dstnodes.end())
    iter++;
  CHECK_STREQ((*iter)->name.c_str(), dst->name.c_str());
  dstnodes.erase(iter);
}

void Node::RemoveSrcNode(Node* src) {
  auto iter = srcnodes.begin();
  while ((*iter)->name != src->name && iter != srcnodes.end())
    iter++;
  CHECK((*iter)->name == src->name);
  srcnodes.erase(iter);
}

/*************************Graph****************************/
Graph::~Graph() {
  for (Node* node : nodes_)
    delete node;
}

void Graph::AddNode(Node* node) {
  nodes_.push_back(node);
  name2node_[node->name] = node;
}

Node* Graph::AddNode(const string& name) {
  Node* node = new Node(name);
  AddNode(node);
  return node;
}

void Graph::AddEdge(Node* srcnode, Node* dstnode) {
  srcnode->AddDstNode(dstnode);
  dstnode->AddSrcNode(srcnode);
}

void Graph::AddEdge(const string& src, const string& dst) {
  CHECK(name2node_.find(src) != name2node_.end())
    <<"can't find src node " << src;
  CHECK(name2node_.find(dst) != name2node_.end())
    <<"can't find dst node " << dst;

  Node* srcnode = name2node_[src], *dstnode = name2node_[dst];
  AddEdge(srcnode, dstnode);
}

void Graph::RemoveEdge(Node* src, Node* dst) {
  src->RemoveDstNode(dst);
  dst->RemoveSrcNode(src);
}

void Graph::RemoveEdge(const string &src, const string& dst) {
  CHECK(name2node_.find(src) != name2node_.end())
    <<"can't find src node " << src;
  CHECK(name2node_.find(dst) != name2node_.end())
    <<"can't find dst node " << dst;

  Node* srcnode = name2node_[src], *dstnode = name2node_[dst];
  RemoveEdge(srcnode, dstnode);
}

const string Graph::ToJson() const {
  map<string, string> info;
  return ToJson(info);
}

const string Graph::ToJson(const map<string, string>& info) const {
  map<string, int> nodeid;
  string disp = "{\"directed\":1,\n";

  // add nodes
  disp += "\"nodes\":[\n";
  bool first = true;

  vector<string> colors = {"red", "blue", "black", "green"};
  // see for more shapes at http://www.graphviz.org/doc/info/shapes.html
  vector<string> shapes = {"box", "ellipse"};
  int id = 0;
  for (auto node : nodes_) {
    char str[1024];
    string name = node->name;
    string color = colors[(node->partition_id)%colors.size()];
    string shape;
    string origin = node->origin;
    if (origin.find("##") != string::npos)
      shape = shapes[1];
    else
      shape = shapes[0];
    snprintf(str, sizeof(str),
        "{\"id\":\"%s%s\", \"color\":\"%s\",\"shape\":\"%s\"}\n", name.c_str(),
        info.find(name) != info.end() ? info.at(name).c_str() : "",
        color.c_str(), shape.c_str());
    if (!first)
      disp += ",";
    else
      first = false;
    disp += string(str);
    nodeid[name] = id++;
  }
  disp += "]\n,";

  // add edges
  disp += "\"links\":[\n";
  first = true;
  for (auto src : nodes_) {
    for (auto dst : src->dstnodes) {
      char str[1024];
      snprintf(str, sizeof(str),
          "{\"source\":%d, \"target\":%d, \"color\":\"%s\"}\n",
          nodeid[src->name], nodeid[dst->name], "black");
      if (!first)
        disp += ",";
      else
        first = false;
      disp += string(str);
    }
  }
  disp += "]\n";
  return disp+"}";
}

// sort to make `bottom' nodes be placed in the front positions
void Graph::Sort() {
  // nodes to be visited
  std::queue<Node*> visiting_nodes;
  // visited node set
  std::unordered_set<Node*> visited_set;
  // visiting_nodes + visted_set
  std::unordered_set<Node*> visit_set;;
  for (auto node : nodes_) {
    // visit nodes without source nodes firstly
    if (node->srcnodes.size() == 0) {
      visiting_nodes.push(node);
      visit_set.insert(node);
    }
  }
  int n = nodes_.size();
  nodes_.clear();
  while (!visiting_nodes.empty()) {
    auto node = visiting_nodes.front();
    visiting_nodes.pop();
    bool visit = true;
    bool bi_direction = false;
    LOG(ERROR)<<"node: "<<node->name;
    for(auto src: node->srcnodes)
      for (auto src_of_src: src->srcnodes)
        if(strcmp((src_of_src->name).c_str(), (node->name).c_str())==0){
          bi_direction=true;
          break;
        }
    LOG(ERROR)<<"bi-direction: "<<bi_direction;
    if (bi_direction && (node->srcnodes).size() > 1){
        LOG(ERROR)<<"special";  
        auto src =  node->srcnodes.at(0); //check whether src nodes number greater than 1
        if(visited_set.find(src) == visited_set.end()){
          visit=false;
        }
    }
    else{
      for(auto src: node->srcnodes)
        if(visited_set.find(src) == visited_set.end()){
          visit=false;
          break;
        }
    }
/*********************************************
    for (auto src : node->srcnodes) {
      // visit this node only if all srouce nodes have been visited
      if (visited_set.find(src) == visited_set.end()) {
        visit = false;
        break;
      }
    }
*********************************************/
    if (visit) {
      nodes_.push_back(node);
      visited_set.insert(node);
      for (auto dst : node->dstnodes) {
        // queueing the dst node if it is not queued before
        if (visit_set.find(dst) == visit_set.end()) {
          visiting_nodes.push(dst);
          visit_set.insert(dst);
        }
      }
    } else {
      visiting_nodes.push(node);
    }
  }
  for(auto node: nodes_){
    LOG(ERROR)<<"nodes: "<<node->name;
  }
  LOG(ERROR)<<"finish printing nodes ";
  CHECK_EQ(nodes_.size(), n);
}

}  // namespace singa
