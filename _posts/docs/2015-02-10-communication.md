---
layout: post
title: Communication
category : docs
tagline:
tags : [development, documentation, communication]
---
{% include JB/setup %}

Different messaging libraries has different benefits and drawbacks. For instance,
MPI provides fast message passing between GPUs (using GPUDirect), but does not
support fault-tolerance well. On the contrary, systems using ZeroMQ can be
fault-tolerant. But ZeroMQ does not support GPUDirect. The AllReduce function
of MPI is also missing in ZeroMQ which is efficient for data aggregation for
distributed training. In Singa, we provide general messaging APIs for
communication between threads within a process and across processes, and let
users choose the underlying implementation (MPI or ZeroMQ) that meets their requirements.

Singa's messaging library consists of two components, namely the message, and
the socket to send and receive messages. **Socket** refers to a
Singa defined data structure instead of the Linux Socket.
We will introduce the two components in detail with the following figure as an
example architecture.

<figure>
<img src="{{ BASE_PATH }}/assets/image/arch/arch2.png" align="center" width="550px"/>
<img src="{{ BASE_PATH }}/assets/image/arch/comm.png" align="center" width="550px"/>
<figcaption><strong> Fig.1 - Example physical architecture and network connection</strong></figcaption>
</figure>

Fig.1 shows an example physical architecture and its network connection.
[Section-partition server side ParamShard]({{ BASE_PATH }}{% post_url /docs/2015-01-30-architecture %}) has a detailed description of the
architecture. Each process consists of one main thread running the stub and multiple
background threads running the worker and server tasks. The stub of the main
thread forwards messages among threads . The worker and
server tasks are performed by the background threads.

## Message

<figure >
<object type="image/svg+xml" align="center" width="100" data="{{ BASE_PATH }}/assets/image/msg.svg" > Not
supported </object>
<figcaption><strong> Fig.2 - Logical message format</strong></figcaption>
</figure>

Fig.2 shows the logical message format which has two parts, the header and the
content. The message header includes the sender and receiver's ID consisting of
the group ID and the worker/server ID within the group. The stub forwards
messages by looking up an address table based on the receiver's ID.
There are two sets of messages, their message types are:

  * kGet/kPut/kRequest/kSync for messages about parameters

  * kFeaBlob/kGradBlob for messages about transferring feature and gradient
  blobs of one layer to its neighboring layer

There is a target ID in the header. If the message is related to parameters,
the target ID is then the parameter ID. Otherwise the message is related to
layer feature or gradient, and the target ID consists of the layer ID and the
blob ID of that layer. The message content has multiple frames to store the
parameter or feature data.

The API for the base Msg is:

    class Msg{
     public:
      /**
       * Destructor to free memory
       */
      virtual ~Msg()=0;
      /**
       * @param group_id worker/server group id
       * @param id worker/server id within the group
       * @param flag 0 for server, 1 for worker, 2 for stub
       */
      virtual void set_src(int group_id, int id, int flag)=0;
      virtual void set_dst(int group_id, int id, int flag)=0;
      virtual void src_group_id()=0;
      virtual int src_id() const=0;
      virtual int dst_group_id() const=0;
      virtual int dst_id() const=0;
      virtual int src_flag() const=0;
      virtual int dst_flag() const=0;
      virtual void set_type(int type)=0;
      virtual void set_target(int param_id)=0;
      virtual void set_target(int layer_id, blob_id)=0;
      virtual int type() const=0;
      /**
       * @return true if the msg is about parameter, otherwise false
       */
      virtual bool is_param_msg() const=0;
      virtual int param_id() const=0;
      virtual int layer_id() const=0;
      virtual int blob_id() const=0;

      virtual void add_frame(void*, int nBytes)=0;
      virtual int frame_size()=0;
      virtual void* frame_data()=0;
      /**
       * Move the cursor to the next frame
       * @return true if the next frame is not NULL; otherwise false
       */
      virtual bool next_frame()=0;
      virtual int SerializeTo(string* buf);
      virtual int ParseFrom(const string& buf);
    };

## Socket

In Singa, there are two types of sockets, the Dealer Socket and the Router
Socket. The names are from ZeroMQ. All connections are of the same type, i.e.,
Dealer<-->Router. The communication between dealers and routers are
asynchronous. In other words, one Dealer
socket can talk with multiple Router sockets, and one Router socket can talk
with multiple Dealer sockets.

### Base Socket

The basic functions of a Singa Socket is to send and receive messages. The APIs
are:

    class Socket{
     public:
      /**
       * @param args depending on the underlying implementation.
       */
      Socket(void* args);
      /**
       * Send a message to connected socket(s), non-blocking. The message will
       * be deallocated after sending, thus should not be used after calling Send();
       * @param  the message to be sent
       * @param  dst the identifier of the connected socket. By default, it is
       * -1, which means sending this message to all connected sockets.
       * @return 1 for success queuing the message for sending, 0 for failure
       */
      virtual int Send(Msg** msg, int dst=-1)=0;
      /**
       * Receive a message
       * @return a message pointer if success; nullptr if failure
       */
      virtual Message* Receive()=0;
    };

    class Poller{
     public:
      /**
       * Add a socket for polling; Multiple sockets can be polled together by
       * adding them into the same poller.
       */
      void Add(Socket* socket);
      /**
       * Poll for all sockets added into this poller.
       * @param duration stop after this number of milliseconds
       * @return pointer to the socket if it has one message in the receiving
       * queue; nullptr if no message in any sockets,
       */
      Socket* Poll(int duation);
    };

### Dealer Socket

The Dealer socket inherits from the base Socket. In Singa, every Dealer socket
only connects to one Router socket as shown in Fig.1.  The connection is set up
by connecting the Dealer socket to the endpoint of a Router socket.

    class Dealer : public Socket{
     public:
      /**
       * Blocking operation to setup the connection with the router, called
       * only once.
       * @param endpoint identifier of the router. For intra-process
       * connection, the endpoint follows the format of ZeroMQ, i.e.,
       * starting with "inproc://"; in Singa, since each process has one
       * router, hence we can fix the endpoint to be "inproc://router" for
       * intra-process. For inter-process, the endpoint follows ZeroMQ's
       * format, i.e., IP:port, where IP is the connected process.
       * @return 1 connection sets up successfully; 0 otherwise
       /
      int Connect(string endpoint);
      /*
       * Since the Dealer socket connects to only one router, it must send to
       * the connected router, thus the dst argument is useless.
       */
      virtual int Send(Msg** msg, int dst=-1);
      virtual Message* Receive();
    };

### Router Socket

The Router socket inherits from the base Socket. One Router socket connects to
at least one Dealer socket.

    class Router : public Socket{
     /**
      * Blocking operation to setup the connection with dealers.
      * It automatically binds to the endpoint for intra-process communication,
      * i.e., "inproc://router".
      *
      * @param endpoint the identifier for the Dealer socket in other process
      * to connect. It has the format IP:Port, where IP is the host machine.
      * If endpoint is empty, it means that all connections are
      * intra-process connection.
      * @param expected_connections total number of connections. This function
      * exits after receiving this number of connections from dealers or after
      * a timeout (1 minutes).
      * @return number of connected dealers.
      */
      int Bind(string endpoint, int expected_connections);
      virtual int Send(Msg** msg, int dst=-1);
      virtual Message* Receive();
    };

## Implementation

### ZeroMQ

**Why [ZeroMQ](http://zeromq.org/)?** Our previous design used MPI for
communication between Singa processes. But MPI is a poor choice when it comes
to fault-tolerance, because failure at one node brings down the entire MPI
cluster. ZeroMQ, on the other hand, is fault tolerant in the sense that one
node failure does not affect the other nodes. ZeroMQ consists of several basic
communication patterns that can be easily combined to create more complex
network topologies.

<figure>
<img src="{{ BASE_PATH }}/assets/image/msg-flow.png" align="center" width="550px"/>
<figcaption><strong> Fig.3 - Messages flow for ZeroMQ</strong></figcaption>
</figure>

The communication APIs of Singa are similar to the DEALER-ROUTER pattern of
ZeroMQ. Hence we can easily implement the Dealer socket using ZeroMQ's DEALER
socket, and Router socket using ZeroMQ's ROUTER socket.
The intra-process can be implemented using ZeroMQ's inproc transport, and the
inter-process can be implemented using the tcp transport (To exploit the
Infiniband, we can use the sdp transport). Fig.3 shows the message flow using
ZeroMQ as the underlying implementation. The messages sent from dealers has two
frames for the message header, and one or more frames for the message content.
The messages sent from routers have another frame for the identifier of the
destination dealer.

Besides the DEALER-ROUTER pattern, we may also implement the Dealer socket and
Router socket using other ZeroMQ patterns. To be continued.

### MPI

Since MPI does not provide intra-process communication, we have to implement
it inside the Router and Dealer socket. A simple solution is to allocate one
message queue for each socket. Messages sent to one socket is inserted into the
queue of that socket. We create a SafeQueue class to ensure the consistency of
the queue. All queues are created by the main thread and
passed to all sockets' constructor via *args*.

    /**
     * A thread safe queue class.
     * There would be multiple threads pushing messages into
     * the queue and only one thread reading and popping the queue.
     */
    class SafeQueue{
     public:
      void Push(Msg* msg);
      Msg* Front();
      void Pop();
      bool empty();
    };

For inter-process communication, we serialize the message and call MPI's
send/receive functions to transferring them. All inter-process connections are
setup by MPI at the beginning. Consequently, the Connect and Bind functions do
nothing for both inter-process and intra-process communication.

MPI's AllReduce function is efficient for data aggregation in distributed
training. For example, [DeepImage of Baidu](http://arxiv.org/abs/1501.02876)
uses AllReduce to aggregate the updates of parameter from all workers. It has
similar architecture as [Fig.2]({{ BASE_PATH }}{% post_url /docs/2015-01-30-architecture %}),
where every process has a server group and is connected with all other processes.
Hence, we can implement DeepImage in Singa by simply using MPI's AllReduce function for
inter-process communication.

<!--
### Server socket

Each server has a DEALER socket to communicate with the stub in the main
thread via an _in-proc_ socket. It receives requests issued from workers and
other servers, and forwarded by the ROUTER of the stub. Since the requests are forwarded by the
stub, we can make the location of workers transparent to server threads. The
stub records the locations of workers and servers.

As explained previously in the
[APIs]({{ BASE_PATH }}{% post_url /docs/2015-03-20-parameter-management %})
for parameter management, some requests may
not be processed immediately but have to be re-queued. For instance, the Get
request cannot be processed if the requested parameter is not available, i.e.,
the parameter has not been put into the server's ParamShard. The re-queueing
operation is implemented sendings the messages to the ROUTER
socket of the stub which treats the message as a newly arrived request
and queues it for processing.

### Worker socket

Each worker thread has a DEALER socket to communicate with the stub in the main
thread via an _in-proc_ socket. It sends (Get/Update) requests to the ROUTER in
the stub which forwards the request to (local or remote) processes. In case of
the partition of ParamShard of worker side, it may also transfer data with other
workers via the DEALER socket. Again, the location of the other side (a server
or worker) of the communication is transparent to the worker. The stub handles
the addressing.

PMClient executes the training logic, during which it generates GET and UPDATE
requests. A request received at the worker's main thread contains ID of the
PMClient instance. The worker determines which server to send the request based
on its content, then sends it via the corresponding socket. Response messages
received from any of the server socket are forwarded to the in-proc ROUTER
socket. Since each response header contains the PMClient ID, it is routed to
the correct instance.

### Stub sockets

#### ROUTER socket
The main thread has a ROUTER socket to communicate with background threads.

It forwards the requests from workers to background servers. There can be
multiple servers.If all servers maintain the same (sub) ParamShard, then the
request can be forwarded to any of them. Load-balance (like round-robin) can be
implemented in the stub to improve the performance. If each server maintains a
sub-set of the local ParamShard, then the stub forwards each request to the
corresponding server.  It also forwards the synchronization requests from
remote servers to local servers in the same way.

In the case of neural network partition (i.e., model partition), neighbor
layers would transfer data with each other. Hence, the ROUTER would forwards
data transfer requests from one worker to other worker. The stub looks up the
location table to decide where to forward each request.

#### DEALER sockets

The main thread has multiple DEALER sockets to communicate with other
processes, one socket per process. Two processes are connected if one of the
following cases exists:

  * one worker group spans across the two processes;
  * two connected server groups are separated in the two processes;
  * workers and the subscribed servers are separated in the two processes.


All messages in SINGA are of multi-frame ZeroMQ format. The figure above demonstrates different types of messages exchanged in the system.

  1. Requests generated by PMClient consist of the parameter content (which could be empty), followed by the parameter ID (key) and the request type (GET/PUT/REQUEST). Responses received by PMClient are also of this format.
  2. Messages received by the worker's main thread from PMClient instances contain another frame identifying the PMClient connection (or PMClient ID).
  3. Requests originating form a worker and arriving at the server contain another frame identifying the worker's connection (or Worker ID).
  4. Requests originating from another server and arriving at the server have the same format as (3), but the first frame identifies the server connection (or Server ID).
  5. After a PMServer processes a request, it generates a message with the format similar to (3) but with extra frame indicating if the message is to be routed back to a worker (a response message) or to route to another server (a SYNC request).
  6. When a request is re-queued, the PMServer generates a message and sends it directly to the server's front-end socket. The re-queued request seen by the server's main thread consists of all the frames in (3), followed by a REQUEUED frame, and finally by another frame generated by the ROUTER socket identifying connection from the PMServer instance. The main thread then strips off these additional two frames before  forwarding it to another PMServer instance like another ordinary request.

-->
