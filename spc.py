'''auth: cmanzo
   V1: aug-2022; orig.
   V2: sep-2022; bug fixes
   V3: oct-2022; bug fixes
   V4: nov-2022; SharePoint list connection
   V5: feb-2023; Production database connection
   V6: feb-2023; new feature: period selection
   '''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
import warnings
from shareplum import Site, Office365
from shareplum.site import Version
import pyodbc
import pandas as pd
import warnings

print()
# ignore warnings
warnings.simplefilter("ignore")

print('MAKE SURE THE RECIPE IS IN THE SHAREPOINT LIST!')

'''
The variables here used are suited to our needs. 
'''



'''Select recipe'''
recipe = input('Input product ID (e.g. 1ABC123400): ')
today = date.today()

delimitar = input(
    'Type T for total analysis or type P to type in a period: ')
print()

if delimitar == "P":
    print('Please proved the date')
    anio = int(input('Year (e.g.: 2023): '))
    mes = int(input('Month (e.g.: 02): '))
    dia = int(input('Day (e.g.: 27): '))

    periodo = date(anio, mes, dia)
    print(f'The SPC will be made for the item: {recipe}')
    print(f'The timeframe goes from {periodo} to {today}')

else:
    print(
        f'The SPC will use all data available for item: {recipe}')
    print()


'''Connect to production software database and get a SQL query to a dataframe'''
cnxn = pyodbc.connect(
    'DRIVER=database;'
    'DBQ=database;'
    'SERVER=xxx.xxx.xxx.x:port/name_of_db;'
    'UID=username;'
    'PWD=password;')

sql = '''SELECT columns
         FROM table
         WHERE REZEPTID = ?'''

BDE = pd.read_sql_query(sql, cnxn, params=[recipe])

cnxn.close()



'''Connect to SharePoint list'''
authcookie = Office365('https://domain.sharepoint.com',
                       username='user@domain.com',
                       password='password').GetCookies()
site = Site('https://site.sharepoint.com/sites/site/',
            version=Version.v365,
            authcookie=authcookie)
sp_list = site.List('SPC') #name of the list in SharePoint
data = sp_list.GetListItems()


data_df = pd.DataFrame(data[0:])
limits = data_df[['Receta','Densidad',
                  'LSL_Temp', 'USL_Temp',
                  'LSL_Energia', 'USL_Energia',
                  'LSL_Potencia', 'USL_Potencia',
                  'LSL_Masa', 'USL_Masa',
                  'LSL_Mixing', 'USL_Mixing']]

limits['Receta'] = limits['Receta'].astype('string')

getlimits_Receta = limits.index[limits['Receta'].str.contains(recipe)]
limits_Receta = limits.loc[getlimits_Receta]

density = limits_Receta['Densidad'].values[0]

'''If period selected: Convert date-time to only date'''
if delimitar == "P":
    BDE["STARTZEIT"] = pd.to_datetime(BDE["STARTZEIT"]).dt.date
    BDE = BDE[(BDE['STARTZEIT'] > periodo)]

else: 
   BDE


'''Get averages and standard deviation per variable'''
def avg_std(variable):
   meanVar = BDE[f'{variable}'].mean()
   stdVar = BDE[f'{variable}'].std()

   return meanVar, stdVar


meanCiclo, stdCiclo = avg_std('DAUERIST')
meanTemp, stdTemp = avg_std('MAXTEMP')
meanEnergia, stdEnergia = avg_std('SPEZENERGIE')
meanMasa, stdMasa = avg_std('MASSE')
meanPotencia, stdPotencia = avg_std('MAXPOWER')
meanMixing, stdMixing = avg_std('MISCHZEIT')


'''Data cleanup - remove unnecesary data or outliers (due to corrupted data)'''
BDE.drop(BDE[BDE['BAKZ'] == ('N' or 'D')].index, inplace=True)
BDE.drop(BDE[BDE['MASSE'] <= 100].index, inplace=True)

BDE.drop(BDE[BDE['DAUERIST'] < (meanCiclo - 3*stdCiclo)].index, inplace=True)
BDE.drop(BDE[BDE['DAUERIST'] > (meanCiclo + 3*stdCiclo)].index, inplace=True)

BDE.drop(BDE[BDE['MAXTEMP'] < (meanTemp - 3*stdTemp)].index, inplace=True)
BDE.drop(BDE[BDE['MAXTEMP'] > (meanTemp + 3*stdTemp)].index, inplace=True)

BDE.drop(BDE[BDE['SPEZENERGIE'] < (meanEnergia - 3*stdEnergia)].index, inplace=True)
BDE.drop(BDE[BDE['SPEZENERGIE'] > (meanEnergia + 3*stdEnergia)].index, inplace=True)

BDE.drop(BDE[BDE['MASSE'] < (meanMasa - 3*stdMasa)].index, inplace=True)
BDE.drop(BDE[BDE['MASSE'] > (meanMasa + 3*stdMasa)].index, inplace=True)

BDE.drop(BDE[BDE['MAXPOWER'] < (meanPotencia - 3*stdPotencia)].index, inplace=True)
BDE.drop(BDE[BDE['MAXPOWER'] > (meanPotencia + 3*stdPotencia)].index, inplace=True)

BDE.drop(BDE[BDE['MISCHZEIT'] < (meanMixing - 3*stdMixing)].index, inplace=True)
BDE.drop(BDE[BDE['MISCHZEIT'] > (meanMixing + 3*stdMixing)].index, inplace=True)



''' __    ___   ___      ___    ___
   |  \  |     |     |  |   \  |
   |__/  |__   |     |  |___/  |___
   |  \  |     |     |  |      |
   |   \ |___  |___  |  |      |___
'''

'Count instances of product'
instancias_receta = BDE[BDE['REZEPTID'] == recipe]['REZEPTID'].count()

'''Calculate fill factor'''
BDE['FF'] = np.where(BDE['REZEPTID'].str[-1] == 2,
                     BDE['MASSE']/density/190, 
                     BDE['MASSE']/density/320) # nueva columna con factor de llenado


'''Column rename'''
BDE.columns = ['Orden', 'BAKZ', 'REZEPT ID', 'REZEPT BEZ', 'Batch', 
               'HoraInicio', 'Ciclo', 'Temp0', 'Temp', 'Temp_min', 
               'Temp_descarga', 'Energia', 'Vueltas', 'Masa', 'Potencia', 
               'Ciclo_Carga','Dosif', 'Mixing_all', 'Vaciado', 
               'Espera', 'Manual', 'Fegezeit', 'Mixing', 'FF']


'''Create df for global info'''
allReceta = BDE[['Ciclo', 'Temp', 'Energia', 'Masa',
                 'Potencia', 'Mixing', 'Dosif']]


''' __
   |__| |   |
   |  | |__ |__
'''

'''Individual averages and ranges'''
avgAll = allReceta.mean()
stdAll = allReceta.std()
maxAll = allReceta.max()
minAll = allReceta.min()
rangoind = maxAll - minAll


'''Data summary and correlation'''
resumen = BDE.describe()
cor = BDE.corr()


'''Get values out of specs - individual values'''
def fuera3s(variable):
   off_3s_ID = BDE.index[BDE[f'{variable}'] >= (avgAll[f'{variable}'] + 3 * stdAll[f'{variable}'])]
   off_3s = BDE.loc[off_3s_ID]
   return off_3s


off_3sCiclo = fuera3s('Ciclo')
off_3sTemp = fuera3s('Temp')
off_3sEnergia = fuera3s('Energia')
off_3sMasa = fuera3s('Masa')
off_3sPotencia = fuera3s('Potencia')
off_3sMixing = fuera3s('Mixing')


''' __   __   __
   |__  |__| |
    __| |    |__
'''

'''
Constants for ≥ 25 elements
These are used due to the criteria established in our process
'''
A2 = 0.153
d2 = 3.931
D3 = 0.459
D4 = 1.541


'Averages and ranges per order'
promedioSubrgupoAll = BDE.groupby('Orden').mean()
maxSubrgupoAll = BDE.groupby('Orden').max()
minSubrgupoAll = BDE.groupby('Orden').min()

promedioSubrgupo = promedioSubrgupoAll.iloc[:, 1:20]
maxSubrgupo = maxSubrgupoAll.iloc[:, 5:]
minSubrgupo = minSubrgupoAll.iloc[:, 5:]
rangoSubgrupo = maxSubrgupo - minSubrgupo

instancias_ordenes = promedioSubrgupoAll['Batch'].count()


'''
Average of averages
Average of ranges
'''
granPromedio = promedioSubrgupo.mean()
promedioRango = rangoSubgrupo.mean()

desviacionStdX = promedioRango / d2
desviacionStdXSubgrupo= desviacionStdX.std()


'''
Central line
Control limits
'''
CLx = granPromedio
CLr = promedioRango

UCLx = granPromedio + A2*promedioRango
LCLx = granPromedio - A2*promedioRango
UCLr = D4*promedioRango
LCLr = D3*promedioRango


'''
CP & CPk
'''

def capacidad(variable):
   USL = limits_Receta[f'USL_{variable}'].values[0]
   LSL = limits_Receta[f'LSL_{variable}'].values[0]
   
   cp  = (USL - LSL)/(6*desviacionStdX[f'{variable}'])
   cpu = (USL - granPromedio[f'{variable}'])/(3*desviacionStdX[f'{variable}'])
   cpl = (granPromedio[f'{variable}'] - LSL)/(3*desviacionStdX[f'{variable}'])
   cpk = min(cpu, cpl)

   pp = (USL - LSL)/(6*stdAll[f'{variable}'])
   ppu = (USL - granPromedio[f'{variable}'])/(3*stdAll[f'{variable}'])
   ppl = (granPromedio[f'{variable}'] - LSL)/(3*stdAll[f'{variable}'])
   ppk = min(ppu, ppl)

   return USL, LSL, cp, cpu, cpl, cpk, pp, ppu, ppl, ppk


USLTemp, LSLTemp, cpTemp, cpuTemp, cplTemp, cpkTemp, ppTemp, ppuTemp, pplTemp, ppkTemp = capacidad('Temp')
USLEnergia, LSLEnergia, cpEnergia, cpuEnergia, cplEnergia, cpkEnergia, ppEnergia,ppuEnergia, pplEnergia, ppkEnergia= capacidad('Energia')
USLPotencia, LSLPotencia, cpPotencia, cpuPotencia, cplPotencia, cpkPotencia, ppPotencia, ppuPotencia, pplPotencia, ppkPotencia= capacidad('Potencia')
USLMasa, LSLMasa, cpMasa, cpuMasa, cplMasa, cpkMasa, ppMasa, ppuMasa, pplMasa, ppkMasa= capacidad('Masa')
USLMixing, LSLMixing, cpMixing, cpuMixing, cplMixing, cpkMixing, ppMixing, ppuMixing, pplMixing, ppkMixing= capacidad('Mixing')


'''
Values off limits
'''
def offSPC(variable):
   off_SPCvar = promedioSubrgupoAll.index[(promedioSubrgupoAll[f'{variable}'] <= UCLx[f'{variable}']) | (
                                           promedioSubrgupoAll[f'{variable}'] >= LCLx[f'{variable}'])]
   off_SPC = promedioSubrgupoAll.loc[off_SPCvar]

   return off_SPC


off_SPCCiclo = offSPC('Ciclo')
off_SPCTemp = offSPC('Temp')
off_SPCEnergia = offSPC('Energia')
off_SPCMasa = offSPC('Masa')
off_SPCPotencia = offSPC('Potencia')
off_SPCMixing = offSPC('Mixing')


'''
 ___        ____   _____   ____
|   | |    |    |    |    |
|___| |    |    |    |    |____
|     |    |    |    |         |
|     |___ |____|    |     ____|
'''

'''Individual values'''

'Histogramas y boxplots'
fig1, ax1 = plt.subplots(nrows=3, ncols=2, constrained_layout=True)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('font', size=8)
plt.suptitle(f"Normality & outlier detection - {recipe}", fontsize=12)


ax1[0, 0].hist(BDE['Ciclo'], bins=9, edgecolor='black')
ax1[0, 0].set_title('Cycle time',fontsize='small')
ax1[0, 1].boxplot(BDE['Ciclo'])
ax1[0, 1].set_title('Cycle time',fontsize='small')

ax1[1, 0].hist(BDE['Masa'], bins=9, edgecolor='black')
ax1[1, 0].set_title('Batch weight', fontsize='small')
ax1[1, 1].boxplot(BDE['Masa'])
ax1[1, 1].set_title('Cycle time', fontsize='small')

ax1[2, 0].hist(BDE['Mixing'], bins=9, edgecolor='black')
ax1[2, 0].set_title('Mixing time', fontsize='small')
ax1[2, 1].boxplot(BDE['Mixing'])
ax1[2, 1].set_title('Mixing time', fontsize='small')

plt.savefig(f"{recipe}_{str(today)} - Normality & outlier detection_1.png", dpi=500)
plt.show()

fig2, ax2 = plt.subplots(nrows=3, ncols=2, constrained_layout=True)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('font', size=8)
plt.suptitle(f"Normality & outlier detection - {recipe}", fontsize=12)

ax2[0, 0].hist(BDE['Temp'], bins=9, edgecolor='black')
ax2[0, 0].set_title('Max. temp.',fontsize='small')
ax2[0, 1].boxplot(BDE['Temp'])
ax2[0, 1].set_title('Max. temp.',fontsize='small')

ax2[1, 0].hist(BDE['Energia'], bins=9, edgecolor='black')
ax2[1, 0].set_title('Specific energy',fontsize='small')
ax2[1, 1].boxplot(BDE['Energia'])
ax2[1, 1].set_title('Specific energy',fontsize='small')

ax2[2, 0].hist(BDE['Potencia'], bins=9, edgecolor='black')
ax2[2, 0].set_title('Max. power',fontsize='small')
ax2[2, 1].boxplot(BDE['Potencia'])
ax2[2, 1].set_title('Max. power', fontsize='small')

plt.savefig(f"{recipe}_{str(today)} - Normality & outlier detection_2.png", dpi=500)
plt.show()


'Heatmap'
plt.figure()
plt.suptitle(f"Correlation heatmap - {recipe}")
sns.heatmap(cor, annot=True,
            xticklabels=cor.columns,
            yticklabels=cor.columns,
            cmap='coolwarm',
            fmt = '.2f',
            annot_kws={
                'fontsize': 5,
                'fontweight': 'bold',
                'fontfamily': 'serif'})

plt.savefig(f'{recipe}_{str(today)} - Correlation heatmap.png', dpi=500)
plt.show()


'Individual values plot'
x = list(range(instancias_receta))
rowsAll = 3
columnsAll = 1
plt.suptitle(f"Individual batches - {recipe}")


def plotsInd(variable):
   yVar = allReceta[f'{variable}']
   average = [avgAll[f'{variable}']] * instancias_receta
   uStd = [avgAll[f'{variable}'] + 3 * stdAll[f'{variable}']] * instancias_receta
   lStd = [avgAll[f'{variable}'] - 3 * stdAll[f'{variable}']] * instancias_receta
   return yVar, average, uStd, lStd


'Ciclo'
yIDauer, yAvgIDauer, uStdIDauer, lStdIDauer = plotsInd('Ciclo')
plt.subplot(rowsAll, columnsAll, 1)
plt.plot(x, yIDauer, linewidth=1.5)
plt.plot(x, yAvgIDauer, 'darkgoldenrod', uStdIDauer, 'darkslategrey', lStdIDauer, 'darkslategrey')

plt.ylabel("Cycle time")

'Masa'
yMasse, yAvgMasse, uStdMasse, lStdMasse = plotsInd('Masa')
plt.subplot(rowsAll, columnsAll, 2)
plt.plot(x, yMasse, linewidth=1.5)
plt.plot(x, yAvgMasse, 'darkgoldenrod', uStdMasse, 'darkslategrey', lStdMasse, 'darkslategrey')

plt.ylabel("Batch weight")

'Mischzeit'
yMischzeit, yAvgMischzeit, uStdMischzeit, lStdMischzeit = plotsInd('Mixing')
plt.subplot(rowsAll, columnsAll, 3)
plt.plot(x, yMischzeit, linewidth=1.5)
plt.plot(x, yAvgMischzeit, 'darkgoldenrod', uStdMischzeit, 'darkslategrey', lStdMischzeit, 'darkslategrey')

plt.ylabel("Mixing time")

plt.tight_layout()
plt.savefig(f'{recipe}_{str(today)} - Individual batches_1.png', dpi=500)
plt.show()

plt.suptitle(f"Individual batches - {recipe}")

'Temp'
yMTemp, yAvgMTemp, uStdMTemp, lStdMTemp = plotsInd('Temp')
plt.subplot(rowsAll, columnsAll, 1)
plt.plot(x, yMTemp, linewidth=1.5)
plt.plot(x, yAvgMTemp, 'darkgoldenrod', uStdMTemp, 'darkslategrey', lStdMTemp, 'darkslategrey')

plt.ylabel("Temperatures")

'Energia'
ySEnergie, yAvgSEnergie, uStdSEnergie, lStdSEnergie = plotsInd('Energia')
plt.subplot(rowsAll, columnsAll, 2)
plt.plot(x, ySEnergie, linewidth=1.5)
plt.plot(x, yAvgSEnergie, 'darkgoldenrod', uStdSEnergie, 'darkslategrey', lStdSEnergie, 'darkslategrey')

plt.ylabel("Specific Energy")

'Potencia'
yMPower, yAvgMPower, uStdMPower, lStdMPower = plotsInd('Potencia')
plt.subplot(rowsAll, columnsAll, 3)
plt.plot(x, yMPower, linewidth=1.5)
plt.plot(x, yAvgMPower, 'darkgoldenrod', uStdMPower, 'darkslategrey', lStdMPower, 'darkslategrey')

plt.ylabel("Max. Power")

plt.tight_layout()
plt.savefig(f'{recipe}_{str(today)} - Individual batches_2.png', dpi=500)
plt.show()


'''SPC'''
x = list(range(instancias_ordenes))
rows = 2
columns = 1

def plotSPC(variable):
   # Promedios
   yPVar = promedioSubrgupo[f'{variable}']
   uPLimit = [UCLx[f'{variable}']] * instancias_ordenes
   cPLine = [CLx[f'{variable}']] * instancias_ordenes
   lPLimit = [LCLx[f'{variable}']] * instancias_ordenes

   # Rangos
   yRVar = rangoSubgrupo[f'{variable}']
   uRLimit = [UCLr[f'{variable}']] * instancias_ordenes
   cRLine = [CLr[f'{variable}']] * instancias_ordenes
   lRLimit = [LCLr[f'{variable}']] * instancias_ordenes
   
   return yPVar, uPLimit, cPLine, lPLimit, yRVar, uRLimit, cRLine, lRLimit



'Temp'
yPTemp, ul_xTemp, cl_xTemp, ll_xTemp, yRTemp , ul_rTemp , cl_rTemp, ll_rTemp = plotSPC('Temp')
us_xTemp, ls_xTemp = [USLTemp] * instancias_ordenes, [LSLTemp] * instancias_ordenes

#  AVG
plt.suptitle(f"SPC - {recipe} - Temperatura")
plt.subplot(rows, columns, 1)
plt.plot(x, yPTemp, linewidth=1.5)
plt.plot(x, us_xTemp, '#3D4849', ul_xTemp, 'g', cl_xTemp, '#ff6a00', ll_xTemp, 'g', ls_xTemp, '#3D4849')
plt.ylabel("x̄ - Temperature")

# RANGES
plt.subplot(rows, columns, 2)
plt.plot(x, yRTemp, linewidth=1.5)
plt.plot(x, cl_rTemp, '#ff6a00')
plt.plot(x, ul_rTemp, 'g', ll_rTemp, 'g')
plt.ylabel("R - Temperature")

plt.tight_layout()
plt.savefig(f'{recipe}_{str(today)} - SPC Temperatura.png', dpi=500)
plt.show()


'Energia'
yPSpezEnergie, ul_xEnergia, cl_xEnergia, ll_xEnergia, yRSpezEnergie, ul_rEnergia, cl_rEnergia, ll_rEnergia = plotSPC('Energia')
us_xEnergia, ls_xEnergia = [USLEnergia] * instancias_ordenes, [LSLEnergia] * instancias_ordenes

#  AVG
plt.suptitle(f"SPC - {recipe} - Energía")
plt.subplot(rows, columns, 1)
plt.plot(x, yPSpezEnergie, linewidth=1.5)
plt.plot(x, us_xEnergia, '#3D4849', ul_xEnergia, 'g', cl_xEnergia, '#ff6a00', ll_xEnergia, 'g', ls_xEnergia, '#3D4849')
plt.ylabel("x̄ - Specific Energy")

# RANGES
plt.subplot(rows, columns, 2)
plt.plot(x, yRSpezEnergie, linewidth=1.5)
plt.plot(x, cl_rEnergia, '#ff6a00')
plt.plot(x, ul_rEnergia, 'g', ll_rEnergia, 'g')
plt.ylabel("R - Specific Energy")

plt.tight_layout()
plt.savefig(f'{recipe}_{str(today)} - SPC Energía.png', dpi=500)
plt.show()


'Masa'
yPMasse, ul_x_mass, cl_x_mass, ll_x_mass, yRMasse, ul_r_mass, cl_r_mass, ll_r_mass = plotSPC('Masa')
us_x_mass, ls_x_mass = [USLMasa] * instancias_ordenes, [USLMasa] * instancias_ordenes

#  AVGS
plt.suptitle(f"SPC - {recipe} - Masa")
plt.subplot(rows, columns, 1)
plt.plot(x, yPMasse, linewidth=1.5)
plt.plot(x, us_x_mass, '#3D4849', ul_x_mass, 'g', cl_x_mass, '#ff6a00', ll_x_mass, 'g', ls_x_mass, '#3D4849')
plt.ylabel("x̄ - Batch Weight")

# RANGES
plt.subplot(rows, columns, 2)
plt.plot(x, yRMasse, linewidth=1.5)
plt.plot(x, cl_r_mass, '#ff6a00')
plt.plot(x, ul_r_mass, 'g', ll_r_mass, 'g')
plt.ylabel("R - Batch Weight")

plt.tight_layout()
plt.savefig(f'{recipe}_{str(today)} - SPC Masa.png', dpi=500)
plt.show()


'Potencia'
yPMaxPower, ul_xPotencia, cl_xPotencia, ll_xPotencia, yRMaxPower, ul_rPotencia, cl_rPotencia, ll_rPotencia = plotSPC('Potencia')
us_xPotencia, ls_xPotencia = [USLPotencia] * instancias_ordenes, [LSLPotencia] * instancias_ordenes

# AVGS
plt.suptitle(f"SPC - {recipe} - Potencia")
plt.subplot(rows, columns, 1)
plt.plot(x, yPMaxPower, linewidth=1.5)
plt.plot(x, us_xPotencia, '#3D4849', ul_xPotencia, 'g', cl_xPotencia, '#ff6a00', ll_xPotencia, 'g', ls_xPotencia, '#3D4849')
plt.ylabel("x̄ - Max. Power")

# RANGES
plt.subplot(rows, columns, 2)
plt.plot(x, yRMaxPower, linewidth=1.5)
plt.plot(x, cl_rPotencia, '#ff6a00')
plt.plot(x, ul_rPotencia, 'g', ll_rPotencia, 'g')
plt.ylabel("R - Max. Power")

plt.tight_layout()
plt.savefig(f'{recipe}_{str(today)} - SPC Potencia.png', dpi=500)
plt.show()


'Mixing'
yPMischzeit, ul_x_mixing, cl_x_mixing, ll_x_mixing, yRMischzeit, ul_r_mixing, cl_r_mixing, ll_r_mixing = plotSPC('Mixing')
us_x_mixing, ls_x_mixing = [USLMixing] * instancias_ordenes, [LSLMixing] * instancias_ordenes

# AVGS
plt.suptitle(f"SPC - {recipe} - Mixing")
plt.subplot(rows, columns, 1)
plt.plot(x, yPMischzeit, linewidth=1.5)
plt.plot(x, us_x_mixing, '#3D4849', ul_x_mixing, 'g', cl_x_mixing, '#ff6a00', ll_x_mixing, 'g', ls_x_mixing, '#3D4849')
plt.xlabel("Orders")
plt.ylabel("x̄ - Mixing Time")

# RANGES
plt.subplot(rows, columns, 2)
plt.plot(x, yRMischzeit, linewidth=1.5)
plt.plot(x, cl_r_mixing, '#ff6a00')
plt.plot(x, ul_r_mixing, 'g', ll_r_mixing, 'g')
plt.xlabel("Orders")
plt.ylabel("R - Mixing Time")

plt.tight_layout()
plt.savefig(f'{recipe}_{str(today)} - SPC Mixing.png', dpi=500)
plt.show()


'Dosing times'
plt.subplot(2, 2, 1)
plt.suptitle(f"Dosing times - {recipe}")
plt.hist(BDE['Dosif'], bins=9, edgecolor='black')
plt.xlabel("Dosing time")

plt.subplot(2, 2, 2)
plt.boxplot(BDE['Dosif'])
plt.ylabel("Dosing time")
plt.suptitle(f"Dosing times - {recipe}")

yPAbrufzeit, ul_x_dosing, cl_x_dosing, ll_x_dosing, yRAbrufzeit, ul_r_dosing, cl_r_dosing, ll_r_dosing = plotSPC('Dosif')

# AVGS
plt.subplot(2, 2, 3)
plt.plot(x, yPAbrufzeit, linewidth=1.5)
plt.plot(x, cl_x_dosing, 'r', ul_x_dosing, 'g', ll_x_dosing, 'g')
plt.xlabel("Orders")
plt.ylabel("x̄ - Dosing Time")

# RANGES
plt.subplot(2, 2, 4)
plt.plot(x, yRAbrufzeit, linewidth=1.5)
plt.plot(x, cl_r_dosing, 'r')
plt.plot(x, ul_r_dosing, 'g', ll_r_dosing, 'g')
plt.xlabel("Orders")
plt.ylabel("R - Dosing Time")


plt.tight_layout()
plt.savefig(f'{recipe}_{str(today)} - Dosing times.png', dpi=500)
plt.show()


''' ____          ____   ____
   |      \   /  |      |      |
   |____   \ /   |      |____  |
   |       / \   |      |      |
   |____  /   \  |____  |____  |__
'''
'''EXPORT DATA TO EXCEL'''

threeSigma = pd.DataFrame()
threeSigma = pd.concat([off_3sMixing, threeSigma])
threeSigma = pd.concat([off_3sPotencia, threeSigma])
threeSigma = pd.concat([off_3sMasa, threeSigma])
threeSigma = pd.concat([off_3sEnergia, threeSigma])
threeSigma = pd.concat([off_3sTemp, threeSigma])
threeSigma = pd.concat([off_3sCiclo, threeSigma])

recipeSPC = pd.DataFrame()
recipeSPC = pd.concat([BDE, recipeSPC])

analysisSPC = pd.DataFrame()
analysisSPC = pd.concat([rangoSubgrupo, analysisSPC])
analysisSPC = pd.concat([promedioSubrgupo, analysisSPC])

offSPC = pd.DataFrame()
offSPC = pd.concat([off_SPCMixing, offSPC])
offSPC = pd.concat([off_SPCPotencia, offSPC])
offSPC = pd.concat([off_SPCMasa, offSPC])
offSPC = pd.concat([off_SPCEnergia, offSPC])
offSPC = pd.concat([off_SPCTemp, offSPC])
offSPC = pd.concat([off_SPCCiclo, offSPC])

correlations = pd.DataFrame()
correlations = pd.concat([resumen, correlations])
correlations = pd.concat([cor, correlations])

processCapabilitiesData = [['Max. temperature', 
                              cpTemp, cpkTemp, ppTemp, ppkTemp],
                           ['Specific energy', 
                              cpEnergia, cpkEnergia, ppEnergia, ppkEnergia],
                           ['Max. power', 
                              cpPotencia, cpkPotencia, ppPotencia, ppkPotencia],
                           ['Batch weight', 
                              cpMasa, cpkMasa, ppMasa, ppkMasa],
                           ['Mixing time', 
                              cpMixing, cpkMixing, ppMixing, ppkMixing]]
process_Capabilities = pd.DataFrame(processCapabilitiesData,
                                    columns = ['','Cp', 'Cpk', 'Pp', 'Ppk'])


writer = pd.ExcelWriter(f'{recipe}_{str(today)} - SPC.xlsx', engine='xlsxwriter')
workbook = writer.book

recipeSPC.to_excel(writer, sheet_name="Datos")
correlations.to_excel(writer, sheet_name="Correlación")
threeSigma.to_excel(writer, sheet_name="Batches fuera de 3σ")
analysisSPC.to_excel(writer, sheet_name="SPC")
offSPC.to_excel(writer, sheet_name="Órdenes fuera de SPC")
process_Capabilities.to_excel(writer, sheet_name="Capacidad de proceso")

writer.save()
print()
