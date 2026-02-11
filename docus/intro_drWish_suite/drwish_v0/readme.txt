I would like to implement Dr. Eric wish concept how to screen and when to buy stocks in the market.

Note: the code here can access daily and weekly data that were downloaded apriori. main.py is my main file. other dependecies files are situated in /src folder
Tasks:
 1. read Developing Dr. Wish's Trading Models.pdf 
 2. read Developing Dr. Wish's Trading Models.pdf
 2. document more on dr eric wish method
 
 generate plan for implementation. dont generate code:
 
 1. FILE 1: PLan implementation for market regim filtering, GMI. The user input for the GMI calculation can vary: qqq or spy or some other index. Exclude for the moment GMI-P4 and GMI-P6 - I THINK we dont have the posibility to calculate now the breath indicator (can we download it as ticker from yahoo finance?). Also I have no idea how to account for GMI-P6 now. Structure the function is such a way that an inclusion in the future is possible. 
 2. FILE 2: Plan Implementing the GREEN LINE SCREENER.  Instead of actual ATH, consider stocks that are at 10% of ATH (this percetange is a user input). Consider stocks that have rested for a period of 3 months (also, this should be input variable). Look for more ideas on how to implemnt this in the pdf docus.
 4. FILE 3: Plan for csv file that contains the user input variables
 3. For the moment do not plan implementation of blue and black dot screener. 
 
i would like to keep track of GLB candidates, GLB breakouts, GLB breakout failures. GLB candidates - stocks closed to ath that restet for x (user input) period of time, GLB breakout (for each stock latest GLB within latest 5 weeks(user_input)), glb failure breakout - when a breakout  closed 2 consecutive days below GLB. Dr. wish uses monthly timeframes for the detection of GLB. I would like the user to be able to use other timeframes, such as daily, weekly. So, let s say the user would consider a GLB if, i.e. stock in the proximity of ATH and rested for 6 weeks. Both the number (6) and the timeframe (week) should be a user input variable.

 Why Historical Data is Essential for This Method

For this strategy, historical data isn't just helpful—it's the entire foundation of the setup. You can explain it like this:

    "All-Time High" is a Historical Concept: The very first step is finding the highest price a stock has ever reached. You cannot know this by looking at today's price alone. The program must scan the entire available price history to determine this peak value.

    Consolidation is a Measure of Time: The core of the rule is that the stock must "rest" or "consolidate" for at least three months. This requires the program to store and analyze the high prices of, at minimum, the three months immediately following the all-time high to verify that the price stayed below that peak. It's a pattern that unfolds over a period of time, and you need the data for that entire period to recognize it.

In short, the Green Line isn't a value you calculate from a few recent numbers. It's a historical landmark. The program's job is to first find that landmark in the past and then watch to see if the current price is moving beyond it. Without the historical map, there's no landmark to find
 
META had an ATH on 14 february 2025 at  cca $741. Then pull back and retaken the value on 30 iunie. It would have been a GLB, but finnaly closed below it and failed. Then it had another GLB on 31 july 2025. The code must look back in history (1.5 years i.e., user input variable), and lets say display all the GLB candidates, breakouts, failures within a given period (user input variable).  


Dr wric wish has a blog called https://www.wishingwealthblog.com/:

He developed several models:

1. Gmi method for monitoring indexes 
2. Green Line breakout
3. Blue dot
4. black dot

I would like to develop a python project that implements all his techniques, and screens for stocks that have a green line breakout, blue or black dot. Taken the position or not is controled by the overall market; his gmi method I already have acces to downloaded ticker data. I just need to start and build the model. Can you please make a plan for me how to cover all the 4 topics. I have included some references.

https://scan.stockcharts.com/discussion/2477/green-line-breakout-by-eric-wish
https://www.youtube.com/watch?v=dl7lGdHuX2U&t=3001s&pp=ygUSZHIgd2lzaCB0cmFkZXJsaW9u

https://www.tradingview.com/script/qfNJJ8Da-Green-Line-Breakout-GLB-Public-Use/

// This source code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © GreatStockMarketer

//@version=4
study("Green Line Breakout", overlay = true)
int data_range = 470
    
float[] monthHighs = array.new_float(data_range, 0)
float[] GLValues = array.new_float(3, 0)
//float monthHigh = 0.0
float curentGLV = 0.00
float lastGLV = 0.00
int counter = 0
float val = na

//mnthHighValue(period) => security(syminfo.tickerid,"M",high[period],lookahead=true)

for _i = 0 to data_range-1
    array.set(id=monthHighs, index=_i, value=high[data_range-_i])
    //val := mnthHighValue(10)
    
for _j = 0 to data_range-1
    if array.get(id=GLValues, index=0) < array.get(id=monthHighs, index=_j)
        curentGLV := array.get(id=monthHighs, index=_j)
        array.set(id=GLValues, index=0,value=curentGLV) //Holds the current GLV
        counter := 0
    
    if array.get(id=GLValues, index=0) > array.get(id=monthHighs, index=_j)
        counter := counter + 1
        
        if counter == 3 //and ((month(time) != month(timenow)) or (year(time) != year(timenow)))
            lastGLV := array.get(id=GLValues, index=0)
            array.set(id=GLValues, index=1,value=lastGLV) //Holds the current GLV
            counter=0

if timeframe.ismonthly == false
    array.set(id=GLValues, index=1,value=na)
plot(array.get(GLValues,1), color=timeframe.ismonthly?color.green:color.white, trackprice=true, show_last=1, linewidth=3)



https://www.tradingview.com/script/Daql0Dxr-Blue-Dot/

//@version=5
indicator("Blue Dot", shorttitle="Blue Dot", overlay=true)

// Stochastic calculation (10-period)
k = ta.stoch(close, high, low, 10)

// 50-period simple moving average
avgc50 = ta.sma(close, 50)

longSignal = k[1] < 20 and k > 20 and avgc50 > avgc50[1]

// Plot a dot directly on the price chart
plotchar(longSignal, title="Blue Dot", location=location.belowbar, 
          char="•", size=size.small, color=color.blue)

alertcondition(longSignal, title="Stochastic Breakout Signal", message="Blue Dot triggered")

https://usethinkscript.com/threads/dr-wishs-black-green-dot-indicator-for-thinkorswim.8110/





