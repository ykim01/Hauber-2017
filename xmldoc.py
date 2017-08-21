from lxml import etree

#load in xml file from desktop
#tree = etree.parse("http://exist.scta.info/exist/apps/scta-app/document/pl-critical/l1d1c1/transcription")
tree = etree.parse("/home/ykim/Desktop/pl-l1d1c1.xml")

#print(tree.xpath(".//text()"))
#get all paragraphs in document
paragraphs = tree.xpath("//tei:p[contains(., 'divnas')]", namespaces= {"tei" : "http://www.tei-c.org/ns/1.0"})

print(len(paragraphs))

#print the text for paragraph 5
print(paragraphs[0].text)
