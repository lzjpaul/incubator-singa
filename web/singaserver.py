import os
import sys
import json
import shutil
import glob
import subprocess
import zipfile
from flask import Flask, request, make_response, send_from_directory
from werkzeug import secure_filename

ALLOWED_EXTENSIONS = set(['zip'])
job_dir = 'static/job/'
upload_dir = 'static/upload/'
singa_dir = '.'

DEMO = True

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

def allowed_file(filename):
  return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route("/api/upload", methods = ['GET', 'POST'])
def upload():
  if request.method == 'POST':
    file = request.files['file']
    if file and allowed_file(file.filename):
      filename = secure_filename(file.filename)
      file.save(os.path.join(upload_dir, filename))
      zf = zipfile.ZipFile(os.path.join(upload_dir, filename), 'r')
      jobid = -1
      for f in zf.namelist():
        if f.endswith('.log'):
          jobid = int(os.path.splitext(os.path.split(f)[-1])[0])
          break
      if jobid < 0:
        return json.dumps({'result': 'error', 'data': 'cannot find correct files from uploaded zip %s' % filename})
      idstr = 'job' + str(jobid)
      cur_job_dir = os.path.join(job_dir, idstr)
      if os.path.isdir(cur_job_dir):
        print 'delete previous folder %s' % cur_job_dir
        shutil.rmtree(cur_job_dir)
      for f in zf.namelist():
        zf.extract(f, cur_job_dir)
        if f.endswith('.log'):
          records = []
          with open(os.path.join(cur_job_dir, f), 'r') as fd:
            records=fd.readlines()
          Joblist[idstr]={'idx':0, 'records': records}
      print(len(records))
      return json.dumps({'result': 'success', 'data': {'jobid': str(jobid), 'num':len(records)}})
    else:
      return json.dumps({'result': 'error', 'data': 'error in uploading file'})
  else:
    return json.dumps({'result': 'error', 'data': 'must use POST'})

@app.route("/api/download/<jobid>", methods = ['GET'])
def download(jobid):
  idstr = 'job' + str(jobid)
  cur_job_dir = os.path.join(job_dir, idstr)
  if not os.path.isdir(cur_job_dir): #idstr not in Joblist or
    return json.dumps({'result': 'error', 'data': 'not such job %s' % jobid})
  else:
    zfpath = cur_job_dir.strip('/') + '.zip'
    zf = zipfile.ZipFile(zfpath, mode = 'w')
    try:
      for f in os.listdir(cur_job_dir):
        fpath = os.path.join(cur_job_dir, f)
        if os.path.isfile(fpath):
          zf.write(fpath, f)
    finally:
      zf.close()
    return json.dumps({'result':'success', 'data':{'url': zfpath}})

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
  jobid = -1
  for line in output:
    if 'jobid' in line:
      jobid = int(line.split('job_id =')[1].split(']')[0])
      break
  assert jobid >= 0, 'Wrong job id %d' % jobid
  idstr = 'job' + str(jobid)
  if idstr not in Joblist:
    Joblist[idstr] = [procs, output, workspace]
    cur_job_dir = os.path.join(job_dir, idstr)
    if os.path.isdir(cur_job_dir):
      print 'delete previous job folder %s ' % cur_job_dir
      shutil.rmtree(cur_job_dir)
    os.makedirs(cur_job_dir)
  else:
    return json.dumps({'result':'error', 'data': 'Repeated job ID'})
  return json.dumps({'result': 'success', 'data': {'jobid': jobid}})

@app.route("/api/get/<jobid>", methods = ['GET'])
def get_record(jobid):
  idstr = 'job' + jobid
  if idstr not in Joblist:
    return json.dumps({'result':'error', 'data':'No such job with id %s' % jobid})
  else:
    idx = Joblist[idstr]['idx']
    if idx < len(Joblist[idstr]['records']):
      Joblist[idstr]['idx'] = idx + 1
      return json.dumps({'result':'success', 'data': [json.loads(Joblist[idstr]['records'][idx])]})
    else:
      return json.dumps({'result': 'success', 'data': []})

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
    with open(os.path.join(job_dir, idstr, jobid + '.log'), 'a') as fd:
      for chart in charts:
        fd.write(json.dumps(chart) + '\n')
    return json.dumps({'result': 'success', 'data': charts})

def poll_image_record(idstr):
  vis_dir = os.path.join(Joblist[idstr][2], 'visualization')
  charts = []
  if not os.path.isdir(vis_dir):
    return charts
  bins = [f for f in os.listdir(vis_dir) if 'step' in f and '.bin' in f]
  for f in bins:
    plot = False
    cur_job_dir = os.path.join(job_dir, idstr)
    if f.endswith('param.bin'):
      plot_param.plot_all_params(os.path.join(vis_dir, f), cur_job_dir)
      plot = True
    elif f.endswith('feature.bin'):
      plot_feature.plot_all_feature(os.path.join(vis_dir, f), cur_job_dir)
      plot = True
    if plot:
      prefix = os.path.splitext(f)[0]
      #the file name is stepxxx-workerxxx-feature|param.bin
      step = f.split('-')[0][4:]
      newimgs = [img for img in os.listdir(cur_job_dir) if img.startswith(prefix)]
      for img in newimgs:
        #the file name is stepxxx-workerxxx-feature|param-<name>.png
        title = os.path.splitext(img)[0].split('-')[-1]
        charts.append({'type' : 'pic', 'step': step, 'title' : title, 'url' : os.path.join(cur_job_dir, img)})
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
  if len(sys.argv) == 1:
    print 'running in demo mode (will not run singa program)'
  elif len(sys.argv) == 2:
    singa_dir = sys.argv[1]
    assert singa_dir, "%s is not a dir " % singa_dir
    assert os.path.isfile(os.path.join(singa_dir, 'bin/singa-run.sh'))
    sys.path.append(os.path.join(singa_dir, 'tool'))
    import plot_param
    import plot_feature
  else:
    print 'Usage: run demo mode with $>python webserver.py ', \
        ' run server mode with $>python webserver.py SINGA_ROOT'
    sys.exit()

  mydir = os.path.split(sys.argv[0])[0]
  job_dir = os.path.join(mydir, job_dir)
  upload_dir = os.path.join(mydir, upload_dir)
  if not os.path.isdir(job_dir):
    os.makedirs(job_dir)
  if not os.path.isdir(upload_dir):
    os.makedirs(upload_dir)


  app.run(host = '0.0.0.0', debug = True, use_debugger = True)
