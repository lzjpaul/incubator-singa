---
layout: post
title: Parameter Management
category : docs
tagline:
tags : [development, documentation, parameter]
---
{% include JB/setup %}

In this article, we describe the parameter management in SINGA.

## Base Classes (Abstractions)

### Param class

Parameters in SINGA are represented by Param objects. (todo: add description for Param class).


### ParameterManager class

ParameterManager (PM) is responsible for synchronizing Param objects between workers
and parameter server.


**Draft**: the PM provides APIs for both workers and servers to get and update Param objects.

        /**
         * block operation called by worker to get the parameter.
         */
        bool Get(Param*);
        /**
         * Processes Get request and responses to the sender.
         * Returns true if success, otherwise returns false.
         */
        bool HandleGet(int paramId, msg_t* msg);
        /**
         * Non-blocking opeartion. It passes the parameter to the PM that maintains it.
         */
        bool Put(Param*);
        /**
         * Non-blocking operation to processes Put request.
         * Returns true if success, otherwise returns false.
         */
        bool HandlePut(int paramId, msg_t* msg);
        /**
         * Non-blocking operation for updating parameters.
         * It may synchronize the updates to other PMs;
         */
        bool Update(Param*);
        /**
         * Blocking operation to collect results from the previous update.
         * E.g., receive responses from remote PM. If the HandleUpdate do not
         * response automatically, it should request explicitly to the PM.
         */
        bool Collect(Param*);
        /**
         * Processes Update requests.
         * It may return responses, e..g, parameter back to the sender.
         */
        bool HandleUpdate(paramId, msg_t* msg);
        /**
         * Returns the node Id on which to send the Get/Put/Update request.
         */
        int Shard(int paramId);
        /**
         * Returns whether to synchronize updates to other PMs.
         */
        bool SyncNow(int paramId);

  With this general PM design, we can support different cluster topologies by
  implementing different PMs. The following figure shows the logical components of three topologies.
  In fact, one server may consist of multiple nodes.

  <img src="{{ BASE_PATH }}/assets/image/history/pm-topology.png" align="center" width="500px"/>

  * Worker-Server. This is the current topology we are using.

  * Worker-Replicated Server. This topology is to reduce the communication overload of a single server (group).

  * Worker-Worker. This topology does not have any parameter servers.


## Worker-Server Architecture

In this section, we describe our implementations for the worker-server architecture.
Workers and servers are multi-threaded. We use ZeroMQ for communication.

**Why ZeroMQ?** Our previous design used MPI for communication between SINGA
processes. But MPI is a poor choice when it comes to failure recovery, because
failure at one node brings down the entire MPI cluster. ZeroMQ, on the other
hand, is fault tolerant in the sense that one node failure does not affect
the other nodes. ZeroMQ consists of several basic communication patterns
that can be easily combined to create more complex network topologies.

### Overview

![Figure 1](http://www.comp.nus.edu.sg/~dinhtta/param_server.jpg)

Both workers and servers consist of one main thread and multiple background threads. The main thread sets up communication sockets with other processes and with the background threads. Its main job is to forward messages between different sockets. The worker and server logic are performed by the backgrounds threads.

### Server sockets

The server has one front-end ROUTER socket to receive requests from the workers and other servers. ROUTER socket is used because it allows for responses messages to be routed back to the correct recipients.

Server background threads are called **PMServers**. They connect to the main thread via an _in-proc_ DEALER socket. When the server receives a request, it distributes the request fairly to one of the background threads by simply forwarding to the DEALER socket. This DEALER-DEALER pattern implements a simple form of load balancing.

As explained previously in the [APIs](http://www.comp.nus.edu.sg/~dbsystem/singa/development,%20documentation/2015/03/12/parameter-management/) for parameter management, depending on the consistency model some requests may not be processed immediately but have to be re-queued. Such re-queueing is implemented by having each PMServer instance connects directly to the front-end ROUTER socket. More specifically, when the PMServer decides that a message has to be requeued, it sends the messages to the front-end ROUTER socket which treats the message as another request arriving form the network and queues it for processing.

A server communicates with another server, for example in the replicated server architecture, by having  DEALER sockets connecting to each of its neighbors' front-end sockets. Note that we opt for DEALER-ROUTER  instead of ROUTER-ROUTER pattern to avoid complexity caused by the handshake protocol between ROUTER sockets. Furthermore, the request-reply pattern supported by DEALER-ROUTER is sufficient to implement synchronization between servers.

### Worker sockets

Each worker connects to a server via a DEALER socket enabling the request-reply communication pattern initiated by the worker. The main thread binds an _in-proc_ ROUTER socket to which
background threads (called **PMClients**) are connected. Note that the internal communication pattern here is DEALER-ROUTER, as opposed to DEALER-DEALER used in the server, because each PMClient must receive the correct response for each request it sent.

PMClient executes the training logic, during which it generates GET and UPDATE requests. A request received at the worker's main thread contains ID of the PMClient instance. The worker determines which server to send the request based on its content, then sends it via the corresponding socket. Response messages received from any of the server socket are forwarded to the in-proc ROUTER socket. Since each response header contains the PMClient ID, it is routed to the correct instance.


### Message Formats

  <img src="http://www.comp.nus.edu.sg/~dinhtta/messages.jpg?1222259157.415" alt="">

All messages in SINGA are of multi-frame ZeroMQ format. The figure above demonstrates different types of messages exchanged in the system.

  1. Requests generated by PMClient consist of the parameter content (which could be empty), followed by the parameter ID (key) and the request type (GET/PUT/REQUEST). Responses received by PMClient are also of this format.
  2. Messages received by the worker's main thread from PMClient instances contain another frame identifying the PMClient connection (or PMClient ID).
  3. Requests originating form a worker and arriving at the server contain another frame identifying the worker's connection (or Worker ID).
  4. Requests originating from another server and arriving at the server have the same format as (3), but the first frame identifies the server connection (or Server ID).
  5. After a PMServer processes a request, it generates a message with the format similar to (3) but with extra frame indicating if the message is to be routed back to a worker (a response message) or to route to another server (a SYNC request).
  6. When a request is re-queued, the PMServer generates a message and sends it directly to the server's front-end socket. The re-queued request seen by the server's main thread consists of all the frames in (3), followed by a REQUEUED frame, and finally by another frame generated by the ROUTER socket identifying connection from the PMServer instance. The main thread then strips off these additional two frames before  forwarding it to another PMServer instance like another ordinary request.

### Parameter Shard

Background threads at the server have access to a shared list of parameter objects (or parameter shard) maintained by a ParamShard object. When processing a request message, PMServer first looks at the request type, then invokes a method from ParamShard according to the request type. PMServer transfers ownership of the message to the ParamShard method.

Each ParamShard method takes as input an ID and the frame containing the request content. It retrieves the Param object with the specified ID, then invokes the handle provided by the Param object. Consistency models are implemented by Param objects.

The ParamShard APIs are similar to that of Param, except for the extra parameter specifying the ID. Additionally, ParamShard APIs contain a method for determining if a Param object needs to be synchronized with other servers:

    bool sync_now(int paramID);


###  Topology

Setting a different network topology for workers and servers, as shown in [previous article](http://www.comp.nus.edu.sg/~dbsystem/singa/development,%20documentation/2015/03/12/parameter-management/), is done via a configuration file. In particular, the config file is read into a ProtoBuf message called Topology and is known to all SINGA processes. The Topology message contains multiple ServerConfig, WorkerConfig and ServerSet messages.

    message Topology{
      repeated ServerConfig server = 5; //parameter server network
      repeated WorkerConfig worker = 6;
      repeated ServerSet server_group = 7;
    }

Each ServerConfig message specifies properties of a server process, namely its ID, network address, its neighbor IDs, synchronization interval and the number threads (or PMServer instances). Each WorkerConfig message represents a worker process, containing the worker's global ID, group and local ID (for replicated server architecture), and the number of threads (or PMClient instances). Finally, the ServerSet message represents a server group in the replicated server architecture. Each server group is identified by a group ID and the set of server IDs.

    message ServerConfig{
      required int32 id = 1;
      required string ip = 2;
      required int32 port = 3;
      repeated int32 neighbor = 4; //upstream neighbors
      required int32 sync_interval = 5; //how many update (per Param) before syncing
      required int32 threads = 6;
    }

    message WorkerConfig{
      required int32 global_id = 1; //node id
      required int32 local_id = 2; //id in the group
      required int32 group_id = 3; //ServerSet ID
      required int32 threads = 4;
    }

    message ServerSet{
      required int32 id = 1;
      repeated int32 neighbor = 2; //set of primary server
    }
