from lxml import etree
import re

#load in xml file from desktop
#tree = etree.parse("http://exist.scta.info/exist/apps/scta-app/document/pl-critical/l1d1c1/transcription")
tree = etree.parse("/home/ykim/Desktop/pl-l1d1c1.xml")


#get all paragraphs in document
text = tree.xpath("//tei:p[contains(., 'divinas')]//text()", namespaces= {"tei" : "http://www.tei-c.org/ns/1.0"})



text_nodes = text[0].split(' ')

text_nodes_clean = re.sub("\s+", " ", text[0]).split(' ')

for node in text_nodes_clean:
	print(node)
	print("\n====")

print("word count")
print(len(text_nodes_clean))

#to run python3 paragraph-breakdown.py
