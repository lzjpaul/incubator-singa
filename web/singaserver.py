import os
import sys
import json
import shutil
import glob
import subprocess
from flask import Flask, request, make_response, send_from_directory

singa_dir = '/home/wangwei/program/asf/incubator-singa'
img_dir = 'static/image/'
log_dir = 'static/log/'

# import plot modules
app = Flask(__name__)
Joblist = {}
@app.route("/", methods = ['GET'])
def indexpage():
  return send_from_directory('static', 'index.html')
 
@app.route("/<path:path>", methods = ['GET'])
def homepage(path):
  return send_from_directory('static', path)
 
@app.route("/api/workspace", methods = ['GET'])
def list_workspace():
  '''
  list available examples
  '''
  example_dir = os.path.join(singa_dir, 'examples')
  if not os.path.isdir(example_dir):
    return json.dumps({'result': 'error', 'data': 'cannot found example dir'})

  all_dirs = os.listdir(example_dir)
  dirs = [os.path.join('examples',d) for d in all_dirs if os.path.isdir(os.path.join(singa_dir, 'examples', d))]
  ret = {'type' : 'workspace', 'values' : dirs}
  return json.dumps({'result': 'success', 'data': ret})


@app.route("/api/submit", methods = ['GET','POST'])
def submit_job():
  '''
  submit a job providing workspace and jobconf
  '''
  try:
    workspace = os.path.join(singa_dir, request.form['workspace'])
    conf = request.form['jobconf']
  except Keyerror:
    return json.dumps({'result': 'error', 'data': 'No workspace or jobconf fields'})

  #workspace = os.path.join(singa_dir, 'examples/cifar10')
  if not os.path.isdir(workspace):
    return json.dumps({'result': 'error', 'data': "No such workspace %s on server" % workspace})
  # delete generated files/dir from later run
  vis_dir = os.path.join(workspace, 'visualization')
  if os.path.isdir(vis_dir):
    shutil.rmtree(vis_dir)
  checkpoint_dir =os.path.join(workspace, 'checkpoitn')
  if os.path.isdir(checkpoint_dir):
    shutil.rmtree(checkpoint_dir)
  for f in glob.glob(os.path.join(workspace, 'job.*')):
    os.remove(f)
  with open(os.path.join(workspace, 'job.conf'), 'w') as fd:
    fd.write(conf)

  procs = subprocess.Popen([os.path.join(singa_dir, 'bin/singa-run.sh'), '-workspace=%s' % workspace], stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
  output = iter(procs.stdout.readline, '')
  job_id = -1
  for line in output:
    if 'job_id' in line:
      job_id = int(line.split('job_id =')[1].split(']')[0])
      break
  assert job_id >= 0, 'Wrong job id %d' % job_id
  idstr = 'job' + str(job_id)
  if idstr not in Joblist:
    Joblist[idstr] = [procs, output, workspace]
  else:
    return json.dumps({'result':'error', 'data': 'Repeated job ID'})
  return json.dumps({'result': 'success', 'data': {'jobid': job_id}})

@app.route("/api/poll/<jobid>", methods = ['GET'])
def poll_progress(jobid):
  '''
  poll for one record
  '''

  idstr = 'job' + jobid
  if idstr not in Joblist:
    return json.dumps({'result':'error', 'data':'No such job with id %s' % jobid})
  else:
    # TDOO gen image records if data availabel
    charts = poll_image_record(idstr)
    if len(charts) == 0:
      charts = poll_chart_record(idstr)
      # parse records from log
    Joblist[idstr].extend(charts)
    with open(os.path.join(log_dir, jobid + '.log'), 'a') as fd:
      for chart in charts:
        fd.write(json.dumps(chart))
    return json.dumps({'result': 'success', 'data': charts})

def poll_image_record(idstr):
  vis_dir = os.path.join(Joblist[idstr][2], 'visualization')
  charts = []
  if not os.path.isdir(vis_dir):
    return charts
  bins = [f for f in os.listdir(vis_dir) if 'step' in f and '.bin' in f]
  for f in bins:
    plot = False
    if f.endswith('param.bin'):
      plot_param.plot_all_params(os.path.join(vis_dir, f), img_dir)
      plot = True
    elif f.endswith('feature.bin'):
      plot_feature.plot_all_feature(os.path.join(vis_dir, f), img_dir)
      plot = True
    if plot:
      prefix = os.path.splitext(f)[0]
      #the file name is stepxxx-workerxxx-feature|param.bin
      step = f.split('-')[0][4:]
      newimgs = [img for img in os.listdir(img_dir) if img.startswith(prefix)]
      for img in newimgs:
        #the file name is stepxxx-workerxxx-feature|param-<name>.png
        title = os.path.splitext(img)[0].split('-')[-1]
        charts.append({'type' : 'pic', 'step': step, 'title' : title, 'url' : os.path.join(img_dir, img)})
    os.remove(os.path.join(vis_dir, f))
  return charts

def poll_chart_record(idstr):
  charts = []
  ret = ''
  try:
    for line in Joblist[idstr][1]:
      if 'step-' in line:
        ret = line
        break
  except StopIteration:
    return charts
  if 'step-' in ret:
    fields = ret.split('step-')
  else:
    return charts
  if len(fields) > 1:
    phase = fields[0].split(' ')[-2]
    rest = fields[1].split(',')
    step = rest[0]
    for i in range(len(rest)-1):
      kv = rest[i+1].strip('\n ').split(':')
      charts.append({'type' : 'chart', 'xlabel' : 'step', 'phase': phase.strip(),'ylabel' : kv[0].strip(), 'data': [{'x': step.strip(), 'y': kv[1].strip()}]})
  return charts

@app.route("/api/pollall/<jobid>", methods = ['GET'])
def get_allprogress(jobid):
  '''
  poll all records
  '''
  idstr = 'job' + jobid
  if idstr not in Joblist:
    return json.dumps({'result': 'error', 'data':'No such job with id %s' % jobid})
  else:
    return json.dumps({'result': 'success', 'data': Joblist[idstr][3:]})

@app.route("/api/kill/<jobid>", methods = ['GET'])
def handle_kill(jobid):
  idstr = 'job' + jobid
  if idstr not in Joblist:
    return json.dumps({'result': 'error', 'data':'no such job with id %s' % idstr})
  else:
    # require passwd
    cmd = os.path.join(singa_dir, 'bin/singa-console.sh')
    procs = subprocess.Popen([cmd, 'kill', jobid], stdout = subprocess.PIPE)
    return json.dumps({'result': 'success', 'data': procs.communicate()[0]})

if __name__ == '__main__':
  if len(sys.argv) != 2:
    print 'Usage: python webserver.py SINGA_ROOT'
    sys.exit()
  singa_dir = sys.argv[1]
  sys.path.append(os.path.join(singa_dir, 'tool'))
  import plot_param
  import plot_feature

  mydir = os.path.split(sys.argv[0])[0]
  img_dir = os.path.join(mydir, img_dir)
  log_dir = os.path.join(mydir, log_dir)
  assert singa_dir, "%s is not a dir " % singa_dir
  if not os.path.isdir(img_dir):
    os.makedirs(img_dir)
  if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

  app.run(host = '0.0.0.0', debug = True, use_debugger = True)
