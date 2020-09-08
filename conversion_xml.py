
import base64
# To encode normal string to base-64

    # test = base64.b64encode(bytes(text, 'utf-8'))

    # print(test)

encoded_string = 'PHJvb3Q+CiAgPG5vZGU+aGVsbG88L25vZGU+CiAgPG5vZGU+d29ybGQ8L25vZGU+Cjwvcm9vdD4='

# To decode base64 string

test = base64.b64decode(encoded_string)


## To convert xml string to xml trees to extract data 

import xml.etree.ElementTree as ET
myroot = ET.fromstring(test)


## Example xml data as a string 

data = '''
<metadata>
<food>
    <item name="breakfast">Idly</item>
    <price>$2.5</price>
    <description>
   Two idly's with chutney
   </description>
    <calories>553</calories>
</food>
</metadata>
'''


myroot = ET.fromstring(data)

## To retrieve parent node

print(myroot.tag)

## To check if any dictionaries are present in the parent node:

print(myroot.attrib)

## To retrieve child node:

print(myroot[0].tag)

## To retrieve all child nodes of the root:


print('Child nodes are retrieved as follows :')
print("")
for x in myroot[0]:
    print(x.tag, x.attrib)

## To extract data inbetween child tags: for ex: idly inside tag <item>

print("Extracting text inbetween tags are as follows :")
print('')

for x in myroot[0]:
    print(x.tag,',',x.text)




