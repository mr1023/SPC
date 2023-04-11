# SPC
SPC for a rubber mixing process, focused mainly on the automotive industry. 
Based on the AIAG's 2nd Edition reference manual. 

Process variables used for this SPC were defined internally (by the Process Engineering Department at PTE México). 
Variables include: 
- Mass
- Temperatures
- Power
- Mixing time (effective mixing, when the ram is down)
- Specific energy

For this program to run, it must be connected to the local network for database access, and a SharePoint list was created where the Process Engineers upload and update their expected process limits (these are set and updated by the PE after many observations).

Then the user gives an input of the recipe / product to analyze, indicates if it should analyze the whole data available or just a timeframe.
After confirmation the program will start running on its own, returning first a glimpse of the data and how many points will be analyzed. 

![SPC_input](https://user-images.githubusercontent.com/85533464/231240636-584c0a1f-f35d-438e-9191-702768662091.png)


Then a series of charts / graphs will be displayed: 
1. Normality and outlier detection - For understanding of the nature of the data. In order to get better results, normality mut be assumed (or proved). 

![SPC_Outliers_1](https://user-images.githubusercontent.com/85533464/231241191-ac03be73-573b-4019-921d-96ccef7add2a.png)

![SPC_Outliers_2](https://user-images.githubusercontent.com/85533464/231241212-84493b49-2966-4e04-a661-126581f18615.png)


2. Correlation heatmap - For understanding the interactions between variables within the recipe (and not the rubber group or the industry itself). 

![SPC_Correlation](https://user-images.githubusercontent.com/85533464/231241635-0b6747bf-88e7-4e80-b964-0785a430ce5b.png)


3. Individual batches - To get a whole picture of the variables' behavior, per batch.  

![SPC_Individual_info_1](https://user-images.githubusercontent.com/85533464/231241770-9db155e7-7468-4741-a380-5083bc4da1d2.png)

![SPC_Individual_info_2](https://user-images.githubusercontent.com/85533464/231241783-dce43fee-e665-44cc-8d2b-6c01a1556268.png)


4. SPC charts - Average of averages on top and average of ranges at the bottom, per process variable. 

Temperature

![SPC_Temp](https://user-images.githubusercontent.com/85533464/231242051-62edf078-fcf8-4aef-9dab-5422080ed98a.png)


Energy 

![ESP_Energy](https://user-images.githubusercontent.com/85533464/231242109-177a2916-801c-442c-a89d-df441670206c.png)


Mass (batch weight)

![SPC_Mass](https://user-images.githubusercontent.com/85533464/231242182-3565de57-d7a2-4de0-99f1-68ae2d1f4589.png)


Power

![SPC_Power](https://user-images.githubusercontent.com/85533464/231242276-ed129650-cdc2-4862-b519-592bdc96b685.png)


Mixing time

![SPC_Mixing](https://user-images.githubusercontent.com/85533464/231242342-0b481e18-b77a-41a0-9812-686cfced9d76.png)


5. Dosing times - While this is not a process variable PE can control, it gives an insight on how it might be affecting the mixing process. 

![SPC_Dosing](https://user-images.githubusercontent.com/85533464/231242550-38133873-df17-4c03-b45d-ef02bfd09bdf.png)


6. Excel file as an output - For further data working, or even evidence of analysis. 
    This file includes: 
    * All data analyzed
    * Correlation chart as raw data
    * Any individual batches considered as outliers (outside of ± 3σ)
    * SPC calculations (Average of averages and average of ranges) 
    * Any orders outside of SPC contol limits
    * Process capabilities (Cp, Cpk, Pp and Ppk)
    
   
   ![SPC_output_file](https://user-images.githubusercontent.com/85533464/231243451-6b8243b0-aa46-4cb8-9869-1ba639699dda.png)

 
