#-*- coding: UTF-8 -*-

import xml.sax

class NodeHandler(xml.sax.ContentHandler):
	def __init__(self):
		print("__init__")
		index = 1
		self.key = "test"
		self.all = {}
		self.content = {}
		
	def printall(self):
		print("self:",self.all)
		
	def startElement(self,tag,attributes):
		print("start_tag----------:",tag);
		if(tag=="struct"):
			self.all[tag] = self.content
			self.content.clear()
		#self.CurrentData = tag
		self.key = tag
		#print("self.key:",self.key);
		
		
	def endElement(self,tag):
		print("-------------end_tag ",tag);
		
# 内容事件处理
	def characters(self, content):
		if(self.key=="END"):
			return 0;
		self.content[self.key] = content
		print("self.key:",self.key," content:",content)
		self.key = "END";
		
if ( __name__ == "__main__" ):
	parser = xml.sax.make_parser()
	
	parser.setFeature(xml.sax.handler.feature_namespaces,0)
	
	Handler = NodeHandler()
	parser.setContentHandler( Handler )
	#parser.parse("tlog_fields.xml")
	parser.parse("Movie.xml")
	
	Handler.printall()