#-*- coding: UTF-8 -*-

import xml.sax
import copy

G_list =[];

class NodeHandler(xml.sax.ContentHandler):
	def __init__(self):
		index = 1
		self.key = "Begin"
		self.all = {}
		self.content = []
		
	def printall(self):
			#table_str = str(self.all)	
			with open("xmltables.txt".format(),'w') as f:
				for k,v in self.all.items():
					newlist = "	".join(v)
					print(newlist,file=f)
		
		#for k,v in self.all.items():
		#	print('{key}:{value}'.format(key=k,value=v))

		
	def startElement(self,tag,attributes):
		#print("start_tag----------:",tag);
		key = attributes["name"]
		if(tag=="struct"):
			self.key = key
			#self.all[self.key] = []
			#print("struct=============::",key);
		if(tag=="macrosgroup"):
			self.key = key
			#self.all[self.key] = []
			#print("macrosgroup=============::",key);
		
		#print("key::",key);
		self.content.append(key)
		
	def endElement(self,tag):
		#print("-------------end_tag ",tag)
		if(tag=="struct"):
			self.all[self.key] = copy.deepcopy(self.content)
			#print(">>>>>>self.content.size ",len(self.content))
			self.content.clear()
		if(tag=="macrosgroup"):
			#self.all[self.key] = copy.deepcopy(self.content) #只保留struct，即表，不保留解释宏，所以注掉
			#print(">>>>>>self.content.size ",len(self.content))
			self.content.clear()
		
# 内容事件处理
	#def characters(self, content):
		#print("self.key:",self.key)
		
def main(tablename):
	parser = xml.sax.make_parser()
	parser.setFeature(xml.sax.handler.feature_namespaces,0)
	Handler = NodeHandler()
	parser.setContentHandler( Handler )
	parser.parse("tlog_fields.xml")
	#parser.parse("Movie.xml")
	
	Handler.printall()
	
	G_list =copy.deepcopy( Handler.all.get(tablename) )
	return G_list
		
if ( __name__ == "__main__" ):
	main("")