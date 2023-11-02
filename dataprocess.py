# Id>LBound and Id<UBound
# LBound      UBound       Label  Anomaly
# ----------  ----------   -----  ------------------
# 40,095,567  40,097,097   1      Undervolt 3.3-3.0v
# 40,097,153  40,098,294   2      Undervolt 3.3-2.8v
# 40,098,367  40,100,076   3      Undervolt 3.3-2.6v
# 40,100,246  40,101,445   4      Undervolt 3.3-2.4v
# 40,101,591  40,103,100   5      Undervolt 3.3-2.3v
# 40,103,109  40,104,425   6      Undervolt 3.3-2.2v
# 40,104,903  40,106,126   7      SPI (VCC temp)
# 40,106,323  40,107,682   8      SPI (Clock)
# 40,107,708  40,108,747   9      MCU heat
# 40,108,764  40,109,971  10      MCU heat low volt
# 39,843,957  39,846,256  11      Firmware buffer overflow

data = []
ruta = '' 
c=0
with open ( 'raw_data.txt' , 'r') as f :
    for x in f.readlines():
        if (c!=0):
            datos = x.replace('\n','').split(';')
            id,macAddress ,message_time,logid,funcid,time,energy,cpu = datos
            data.append((id,macAddress ,message_time,logid,funcid,time,energy,cpu))
        c=c+1

# Definir los rangos y etiquetas en listas
LBounds = [40095567, 40097153, 40098367, 40100246, 40101591, 40103109, 40104903, 40106323, 40107708, 40108764, 39843957]
UBounds = [40097097, 40098294, 40100076, 40101445, 40103100, 40104425, 40106126, 40107682, 40108747, 40109971, 39846256]
Labels = ["Undervolt 3.3-3.0v", "Undervolt 3.3-2.8v", "Undervolt 3.3-2.6v", "Undervolt 3.3-2.4v", "Undervolt 3.3-2.3v", "Undervolt 3.3-2.2v", "SPI (VCC temp)", "SPI (Clock)", "MCU heat", "MCU heat low volt", "Firmware buffer overflow"]

# Definir una función para clasificar el dato
def classify_id(id_value):
    for i, (lbound, ubound) in enumerate(zip(LBounds, UBounds)):
        if lbound < id_value < ubound:
            return "Attack",1
    # Si no se encuentra en ningún rango, devolver None o una etiqueta predeterminada
    return "Normal",0

for d in data :
    output=''
    labelString,label=classify_id(int(d[0]))
    tiempo=(d[2].split('T')[1]).replace(':','')
    output=d[1]+','+tiempo+','+d[3]+','+d[4]+','+d[5]+','+d[6]+','+d[7]+','+str(label)+','+labelString
    #print(output)
    with open ( 'raw_data_transform_1.txt' , 'a+') as f :
        f.write(output+'\n')