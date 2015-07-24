import singaserver
import unittest
import json

class SingaServerTest(unittest.TestCase):
  def setUp(self):
    singaserver.app.testing = True
    self.app = singaserver.app.test_client()

  def tearDown(self):
    pass

  def test_listworkspace(self):
    rv = self.app.get('/api/workspace')
    print rv.data

  def test_submit_job(self):
    fd = open('job.conf', 'r')
    confdata = ''.join(fd.readlines())
    fd.close()
    rv = self.app.post('/api/submit', data = dict(workspace='cifar10', jobconf=confdata))
    print rv.data

  def test_poll(self):
    rv = self.app.get('/api/poll/13');
    print rv.data

if __name__ == '__main__':
  unittest.main()
