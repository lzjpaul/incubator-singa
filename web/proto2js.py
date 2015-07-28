import sys

def getComment(section):
  comment=''
  for lno, line in enumerate(section):
    if line.startswith('//'):
      comment+=line[2:]
    else:
      return comment.strip(), section[lno:]

def getEnumField(obj, section):
  comment, section=getComment(section)
  name, value=section[0].split('=')
  return section[1:], obj+".values['"+name.strip()+"']=new EnumValueDef('"+name.strip()+"',"+value+",'"+comment+"');\n"

def processEnumSection(packObj, section):
  comment, section=getComment(section)
  head=section[0]
  assert(head.endswith('{'))
  enumName=head.strip('{').split(' ')[1]
  objName=enumName
  ret="var "+objName+ "=new EnumDef('"+enumName+"','"+comment+"');\n"
  section=section[1:]
  while len(section)>0:
    section, field=getEnumField(objName, section)
    ret+=field
  ret+=packObj+".enums['"+objName+"']="+objName+";\n"
  return ret

def getMessageField(objName, section):
  comment, section=getComment(section)
  fields=section[0].split('=')
  qualifier, typename, name=fields[0].strip().split(' ')
  if len(fields)>2:
    code=fields[1].strip().split('[')[0]
    default=fields[2].strip().split(']')[0]
    if(typename == "string"):
      default=default.split('\"')[1]
    ret=objName+".fields['"+name+"'] = new FieldDef("+code+",'"+name+"','"+qualifier+"','"+typename+"','"+comment+"','"+default+"');\n"
  else:
    code=fields[1].strip().split(';')[0]
    ret=objName+".fields['"+name+"'] = new FieldDef("+code+",'"+name+"','"+qualifier+"','"+typename+"','"+comment+"');\n"
  return section[1:], ret

def processMessageSection(packObj, section):
  comment, section=getComment(section)
  head=section[0]
  assert(head.endswith('{'))
  msgName=head.strip('{').split(' ')[1]
  objName=msgName
  ret="var "+objName+ "=new MessageDef('"+msgName+"','"+comment+"');\n"
  section=section[1:]
  while len(section)>0:
    section, field=getMessageField(objName, section)
    ret+=field
  ret+=packObj+".messages['"+objName+"']="+objName+";\n"
  return ret

if __name__=="__main__":
  inputFilePath = "../src/proto/job.proto"
  outputFilePath = "static/scripts/model.js"
  ret=""
  section=[]
  toCallEnum=False;
  ret="var packsinga=new Package('singa');\n"
  packObj="packsinga"
  callStack=[]
  sectionStack=[]
  for line in open(inputFilePath):
    line=line.strip(' ;\n')
    if line.startswith("package"):
      package=line.split()[-1]
      packObj="pack"+package
      ret="var "+packObj+"=new Package('"+package+"');\n"
    elif line.startswith('//') or "=" in line:
      section.append(line)
    elif line.startswith('}'): # end of one section
      ret+=callStack.pop()(packObj, section)+'\n'
      section=sectionStack.pop()
    elif "{" in line in line: # valid string
      fields=line.split(' ')
      if fields[0]=="enum":
        callStack.append(processEnumSection)
      elif fields[0]=="message":
        callStack.append(processMessageSection)
      sectionStack.append(section[:])
      section=[]
      section.append(line)

  newModel=""
  beginReplace=False
  for line in open(outputFilePath):
    if beginReplace:
      if "//end auto generate" in line:
        newModel+=ret
        beginReplace=False
      else:
        continue

    if "//begin auto generate" in line:
        beginReplace=True

    newModel+=line


  with open(outputFilePath, "w") as fd:
    fd.write(newModel)
    fd.flush()
    fd.close()
