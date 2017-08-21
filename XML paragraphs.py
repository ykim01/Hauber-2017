from lxml import etree
import re

#load in xml file from desktop
#tree = etree.parse("http://exist.scta.info/exist/apps/scta-app/document/pl-critical/l1d1c1/transcription")
tree = etree.parse("/home/ykim/Desktop/pl-l1d1c1.xml")


#get all paragraphs in document
text = tree.xpath("//tei:p//text()", namespaces= {"tei" : "http://www.tei-c.org/ns/1.0"})

print(type(text[0]))


text_nodes = text[0].split(' ')
text_nodes2 = text[1].split(' ')
print(type(text_nodes[0]))



text_nodes_clean = re.sub("\s+", " ", text[0]).split(' ')
text_nodes_clean2 = re.sub("\s+", " ", text[1]).split(' ')


#for node in text_nodes_clean:
#	print(node)
#	print("\n====")

for node in text_nodes_clean2:
        print(node)
        print("\n====")


print("word count")
print(len(text_nodes_clean2))

#to run python3 paragraph-breakdown.py
#this breaks down the very first paragraph into words
