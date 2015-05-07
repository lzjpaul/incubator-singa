---
layout: post
title: System Architecture
category : docs
tagline:
tags : [development, documentation, system architecture]
---
{% include JB/setup %}

We design the architecture of Singa with the consideration to make it general
to unify existing architectures. Then users can easily test the performance of
different architectures and select the best for their models.

## Logical Architecture

<figure>
<img src="{{ BASE_PATH }}/assets/image/arch/logical.png" align="center" width="550px"/>
<figcaption><strong> Fig.1 - Logical system architecture</strong></figcaption>
</figure>

The logical system architecture is shown in Fig.1 with four worker groups and
two server groups. Each worker group runs against a partition of the
training dataset (called data parallelism) to compute the updates (e.g., the
gradients) of parameters of one model replica.  Worker groups run
asynchronously, while workers within one group run synchronously with each
worker computing updates for a partition of the model parameters (called
model parallelism). Each server group maintains one replica of the model
parameters. It receives requests (e.g., Get/Put/Update) from workers and
handles these requests. Server groups synchronize with neighboring server groups
periodically or according to some rules. One worker (or server) group consists
of multiple (user defined number) threads. These threads may resident in one
process or span across multiple processes.

Each worker (or server) group has a ParamShard object, which contains a full
set of Param objects for a model replica. Since the workers (or servers) may
span across multiple processes, this ParamShard may also be partitioned across
multiple processes.  The ParamShards can share the same memory space for
parameter values if they resident in the same process like
[Caffe](http://caffe.berkeleyvision.org/)'s parallel implementation. Sharing
memory could save memory space, but it could also change the training logic and
thus affects the convergence rate.

Each worker (thread) has a PMWorker (abbr for parameter management on workers)
object, e.g., pmw1, pmw2, which calls the Get/Put/Update functions to get
parameters, put parameters and update parameters. These functions may send
requests to servers that use PMServer (abbr for parameter management on
servers), e.g., pms1, pms2, to handle these requests.

## Physical Architecture

In this section, we describe how to configure Singa to generalize the logical
architecture to the physical architectures of existing systems, e.g., Caffe and
Baidu's DeepImage, Google Brain and Microsoft Adam. The architecture
configuration includes:

  * Number of worker threads per worker group and per process, which decides
   the partitioning of worker side ParamShard

  * Number of server threads per server group and per process, which decides
   the partitioning of server side ParamShard

  * Separation of servers and workers in different processes

  * Number of worker groups per server group

  * Topology of server groups, e.g., ring, tree, fully connected, etc.

We put automatic optimization of the configuration as a future feature.

### No Partition of ParamShard

<figure>
<img src="{{ BASE_PATH }}/assets/image/arch/arch1.png" align="center" width="550px"/>
<figcaption><strong> Fig.2 - Physical system architecture without partitioning
ParamShard</strong></figcaption>
</figure>

Fig.2 shows the architecture by configuring three threads per worker group, two
threads per server group, two worker groups and one server group per process.
Worker threads and server threads run as sub-threads. The main thread runs a
loop as a stub to forward messages. For instance, the Get/Put/Update requests
from workers are forwarded to the local servers, while Sync requests from the
local servers are forwarded to remote servers. In this architecture, every
group is fully contained in one process, hence the ParamShard objects is not
partitioned.  If the ParamShards within one process share the same memory space
for parameter values, the training procedure then follows
[Hogwild](http://i.stanford.edu/hazy/hazy/victor/Hogwild/).  If we only
launch process 1, Singa then runs in standalone mode. If we launch multiple
processes, we can connect the server groups to form different topologies,
e.g., ring or tree. Synchronization is conducted via inter-process
communication between neighboring server groups. In
[Caffe's](http://caffe.berkeleyvision.org/) parallel training architecture,
processes are arranged into a ring. Caffe luanches one thread per model
replica, hence it only supports data parallelism. Singa can also support
model parallelism by partition the model replica among multiple worker
threads within one group.

### Partition Server Side ParamShard

<figure>
<img src="{{ BASE_PATH }}/assets/image/arch/arch2.png" align="center" width="550px"/>
<figcaption><strong> Fig.3 - Physical system architecture, partitioning
server ParamShard</strong></figcaption>
</figure>

Fig.3 shows another physical architecture by configuring one worker group per
process, and two processes per server group. Because the server group spans two
processes, the ParamShard of the server group is partitioned across the two
processes. We only show one server groups in the figure. The vertical lines
represent inter-process communication to synchronize server groups if there
are multiple server groups. In process 1, if the update for a parameter that
residents in process 2, then the PMWorker's update request would be sent to
process 2 via inter-process communication. If the parameter is maintained by
process 1, then the update request is sent to pms2 directly via intra-process
communication. The processes for other requests are the same. Baidu's
[DeepImage](http://arxiv.org/abs/1501.02876) system uses one server group and
its ParamShard is partitioned across all processes. Consequently, the stub of each process is
connected with all other processes' stubs for updating parameters. Like Caffe,
it launches only one thread per worker group, thus only support data
parallelism.

### Partition Worker Side ParamShard

<figure>
<img src="{{ BASE_PATH }}/assets/image/arch/arch3.png" align="center" width="550px"/>
<figcaption><strong> Fig.4 - Physical system architecture, partitioning
worker ParamShard</strong></figcaption>
</figure>

The main difference of the architectures in Fig.4 and Fig.3 is that the worker
group is partitioned over two processes in Fig.4. Consequently, the ParamShard
is partitioned across process 1 and process 2. There are two kinds of
parameters to consider:

  * unique parameter: this kind of parameter exists in only one of the
  partitioned ParamShards (due to model partition), and is updated by one
  worker in its host process.  Hence the update is conducted similar to that in
  Fig.2.

  * replicated parameter: this kind of parameter is replicated in all
  ParamShards (due to data partition), and its update is the aggregation of
  updates from all workers.  The aggregation is processed as follows. First the
  first process is selected as the leader process. (Update/Get/Put) requests of
  PMWorkers on other processes are aggregated locally and then forwarded to the
  leader process which handles it in the main thread. *The main thread of the
  leader process creates a PMWorker over the ParamShard it owns. It handles
  the request by calling the correspondingly functions, e.g., Update/Get/Put().
  The responses from the servers are sent to the first worker of all sub ParamShards.*

### Workers and Servers in Different Processes

<figure>
<img src="{{ BASE_PATH }}/assets/image/arch/arch4.png" align="center" width="550px"/>
<figcaption><strong> Fig.5 - Physical system architecture, separating workers
and servers</strong></figcaption>
</figure>

Fig.5 shows an architecture similar to that in Fig.4 except that the servers
and workers are separated into different processes. Consequently all requests
are sent via inter-process communication and handled by remote servers. More
details on this architecture are explained in
[Worker-Server Architecture]({{ BASE_PATH }}{% post_url /docs/2015-03-20-parameter-management %}). This is
the architecture used by Google's Brain and Microsoft's Adam system. It is also
called the Downpour architecture.

## Conclusion

We have shown that Singa's architecture is general enough
to unify the architectures of existing distributed training systems.
Since the rest architectures can be derived similarly as the above four by
setting different architecture configurations, we do not numerate them here.
