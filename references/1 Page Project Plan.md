# ME C231A / EE C220B Project Plan: MPC for ERCOT BESS Bidding Strategy

## *Thibaud Cambronne, Lazlo Samuel Paul, Maximilian Christof, Agustin Nahuel Coppari Hollmann*

## **1\. Introduction & Motivation**

As renewable penetration increases, volatility in electricity prices creates both a challenge for grid stability and an opportunity for arbitrage using Battery Energy Storage Systems (BESS). **Project Goal**: We aim to design a 2 stage control strategy using MPC to maximize revenue by participating in Day-Ahead (DA) and Real-Time (RT) Energy and Ancillary Service (AS) markets, serving as a benchmark against ongoing research in the eCAL lab with Prof. Scott Moura. We will focus our analysis on the ERCOT market (Texas).

## **2\. Overview Project Scope**

We will develop a two-stage optimization controller. Both stages are designed as **convex optimization problems**.

* **Stage 1 \- Day-Ahead (DA) Scheduler:** Runs at 10 am of Day-1 for hours 0 to 23 of Day 0 (the target day).  
  * Determines bids/commitment for DA Energy, Regulation Up, and Regulation Down capacity (other AS ignored at this stage for simplicity), as well first estimate of the RT energy bids

* **Stage 2 \- Real-Time (RT) MPC Controller:** Manage state-of-charge (SoC) and real-time energy bids while enforcing DA commitments.

If time allows, we would additionally want to consider:

* Integration of the 3 remaining DA Ancillary Services: Responsive Reserve Service (RRS), Non-Spinning Reserve (NSRS), and ERCOT Contingency Reserve Service (ECRS).

* Implementation of a basic forecasting model (e.g., LSTM or XGBoost) to replace the Persistence forecast.

## **3\. Goals & Analysis Metrics**

Our primary objective is **Revenue Maximization**. We will evaluate the controller's performance through three specific lenses:

1. **Value of information:** Quantify the "optimality gap" (= performance) between the Persistence Forecast controller and the Perfect Information controller to understand the theoretical upper bound of revenue.

2. **Horizon sensitivity:** Analyze how horizon handling affects performance and end-of-day SoC by comparing:  
   * *Shrinking Horizon:* Horizon that reduces as the day progresses.  
   * *Receding Horizon:* Fixed window of 24h, 48h, and 72h.

3. **(Optional) Robustness to anomalies:** Evaluate performance on "out-of-distribution" days (e.g., grid failures, extreme weather events) to see if the MPC effectively capitalizes on critical price spikes.

## **5\. Team Task Breakdown**

*To ensure all members gain experience with MPC implementation, we will work together on the formulation and implementation of the MPC. Collaboration will be done through GitHub ([repo here](https://github.com/ThibaudCambronne/MPC-BESS-ERCOT)). Data on prices of the different markets can easily be found on ERCOT’s portal.* 

| Member | Primary Responsibility | Learning Goal Focus |
| ----- | ----- | ----- |
| **Thibaud, Lazlo** | **Forecaster:** Module taking as inputs a control strategy (perfect, persistence, LSTM, etc…), a current time of the day, a horizon and returning a forecast between the current time and the end of the horizon | Machine Learning and Data processing |
| **Lazlo** | **DA stage solver:** Module solving the DA stage, and providing bids for the DA markets, as well as a first estimate of the RT bids and the RT SoC | Convex optimization & Solver interfacing |
| **Max, Agustin** | **RT stage solver:** Module solving the RT MPC given a horizon | MPC |
| **All** | **Combine the DA and the RT stages:** Evaluate performance | MPC |
| **Thibaud** | **Horizon sensitivity analysis:** Analyze how horizon handling affects performance and end-of-day SoC | MPC |
| **?** | **(Optional) Robustness to anomalies:** Evaluate performance on "out-of-distribution" days | MPC |

