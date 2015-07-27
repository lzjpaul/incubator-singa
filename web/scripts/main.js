//handlebar helper

Handlebars.registerHelper('trimString50', function(passedString) {
  if(!passedString){
    return new Handlebars.SafeString("");
  }
  var theString = passedString.substring(0,50);
  return new Handlebars.SafeString(theString)
});


