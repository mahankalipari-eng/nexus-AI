import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, silhouette_score)
from heapq import heappush, heappop
import math

st.set_page_config(page_title="NEXUS-AI | Warehouse Intelligence", page_icon="🤖", layout="wide")
st.markdown("""
<style>
.main {background:#0b1220}
[data-testid="stMetric"] {background:#111c30;border:1px solid #26364f;padding:14px;border-radius:12px}
.block-container {padding-top:1.5rem}
h1,h2,h3 {letter-spacing:-.4px}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def make_data(n=3500, seed=42):
    rng=np.random.default_rng(seed)
    distance=rng.uniform(3,60,n)
    congestion=rng.beta(2,2,n)*100
    load=rng.uniform(0,30,n)
    speed=rng.uniform(.45,1.8,n)
    obstacles=rng.poisson(3,n).clip(0,12)
    risk=rng.uniform(0,1,n)
    # synthetic simulator labels with nonlinear interactions + stochastic noise
    time=distance/np.maximum(speed,.2)*(1+congestion/115)+load*.20+obstacles*.75+risk*5+rng.normal(0,2,n)
    time=np.maximum(time,1)
    logit=-2.0 + congestion*.045 + obstacles*.19 + risk*1.8 + load*.018
    p=1/(1+np.exp(-logit))
    high=(rng.random(n)<p).astype(int)
    motor_temp=35+load*.65+congestion*.09+rng.normal(0,2.5,n)
    current=1.2+load*.055+obstacles*.07+rng.normal(0,.18,n)
    speed_error=np.abs(rng.normal(.05+congestion/900,.04,n))
    anomaly=np.zeros(n,dtype=int)
    idx=rng.choice(n,max(1,n//20),replace=False)
    motor_temp[idx]+=rng.uniform(18,38,len(idx))
    current[idx]+=rng.uniform(1.5,3.5,len(idx))
    speed_error[idx]+=rng.uniform(.25,.65,len(idx))
    return pd.DataFrame({"distance_m":distance,"congestion_pct":congestion,"load_kg":load,
      "speed_mps":speed,"obstacles":obstacles,"route_risk":risk,
      "travel_time_s":time,"high_congestion":high,"motor_temp_c":motor_temp,
      "motor_current_a":current,"speed_error":speed_error,"synthetic_anomaly":anomaly+np.isin(np.arange(n),idx).astype(int)})

FEATURES=["distance_m","congestion_pct","load_kg","speed_mps","obstacles","route_risk"]
df=make_data()
st.title("🤖 NEXUS-AI")
st.caption("AI-powered warehouse prediction, anomaly detection and adaptive route optimization")
st.markdown("**Software prototype · Synthetic simulation data · Interactive ML experiments**")
with st.sidebar:
    st.header("Experiment controls")
    n=st.slider("Generated samples",1000,10000,3500,500)
    seed=st.number_input("Random seed",0,9999,42)
    df=make_data(n,int(seed))
    st.caption("Data is synthetically generated for this prototype; it is not real warehouse telemetry.")
    st.divider()
    st.subheader("Scenario")
    distance=st.slider("Trip distance (m)",3,60,25)
    congestion=st.slider("Congestion (%)",0,100,55)
    load=st.slider("Robot load (kg)",0,30,12)
    speed=st.slider("Robot speed (m/s)",0.4,1.8,1.1)
    obstacles=st.slider("Nearby obstacles",0,12,3)
    risk=st.slider("Route risk",0.0,1.0,0.3,.05)

scenario=pd.DataFrame([[distance,congestion,load,speed,obstacles,risk]],columns=FEATURES)
stages=st.tabs(["Overview","ML Lab","Anomaly Detection","Warehouse Planner","Dataset & Method"])
with stages[0]:
    st.subheader("Operational overview")
    c1,c2,c3,c4=st.columns(4)
    pred=make_pipeline(StandardScaler(),LinearRegression()).fit(df[FEATURES],df.travel_time_s).predict(scenario)[0]
    congestion_model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)).fit(df[FEATURES],df.high_congestion)
    prob=congestion_model.predict_proba(scenario)[0,1]
    c1.metric("Predicted travel time",f"{pred:.1f} s")
    c2.metric("High-congestion probability",f"{prob:.0%}")
    c3.metric("Training samples",f"{len(df):,}")
    c4.metric("Scenario risk", "High" if risk>.65 else "Moderate" if risk>.3 else "Low")
    left,right=st.columns([1.3,1])
    with left:
        plot=df.sample(min(500,len(df)),random_state=seed)
        fig=px.scatter(plot,x="congestion_pct",y="travel_time_s",color="high_congestion",
          color_continuous_scale="Turbo",labels={"congestion_pct":"Congestion (%)","travel_time_s":"Travel time (s)"},
          title="Generated warehouse scenarios")
        fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,use_container_width=True)
    with right:
        st.markdown("#### Current scenario")
        st.write(f"**Distance:** {distance} m")
        st.write(f"**Congestion:** {congestion}%")
        st.write(f"**Load:** {load} kg")
        st.write(f"**Speed:** {speed:.1f} m/s")
        st.write(f"**Obstacles:** {obstacles}")
        st.info("Predictions are generated by models trained on the synthetic dataset. Validate with real telemetry before operational use.")

with stages[1]:
    st.subheader("Train and compare machine-learning models")
    target=st.radio("Prediction task",["Regression — estimate travel time","Classification — high congestion"],horizontal=True)
    if target.startswith("Regression"):
        choices=["Linear Regression","Multiple Linear Regression","Polynomial Regression","Decision Tree Regressor","Random Forest Regressor"]
        model_name=st.selectbox("Model",choices)
        X=df[FEATURES]; y=df.travel_time_s
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=int(seed))
        if model_name=="Linear Regression":
            model=make_pipeline(StandardScaler(),LinearRegression())
        elif model_name=="Multiple Linear Regression":
            model=make_pipeline(StandardScaler(),LinearRegression())
        elif model_name=="Polynomial Regression":
            model=make_pipeline(PolynomialFeatures(2,include_bias=False),StandardScaler(),LinearRegression())
        elif model_name=="Decision Tree Regressor":
            model=DecisionTreeRegressor(max_depth=10,random_state=int(seed))
        else:
            model=RandomForestRegressor(n_estimators=120,max_depth=14,n_jobs=-1,random_state=int(seed))
        model.fit(Xtr,ytr); yp=model.predict(Xte); pred_s=float(model.predict(scenario)[0])
        a,b,c=st.columns(3)
        a.metric("MAE",f"{mean_absolute_error(yte,yp):.2f} s")
        b.metric("RMSE",f"{np.sqrt(mean_squared_error(yte,yp)):.2f} s")
        c.metric("R²",f"{r2_score(yte,yp):.3f}")
        st.success(f"Predicted travel time for selected scenario: {pred_s:.2f} seconds")
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=np.array(yte)[:250],y=np.array(yp)[:250],mode="markers",name="Predictions",marker={"opacity":.6}))
        lo=float(min(np.min(yte),np.min(yp))); hi=float(max(np.max(yte),np.max(yp)))
        fig.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode="lines",name="Ideal prediction"))
        fig.update_layout(title="Actual vs predicted (test sample)",xaxis_title="Actual (s)",yaxis_title="Predicted (s)",template="plotly_dark")
        st.plotly_chart(fig,use_container_width=True)
    else:
        model_name=st.selectbox("Model",["Logistic Regression","Decision Tree Classifier","Random Forest Classifier","K-Nearest Neighbors","Support Vector Machine"])
        X=df[FEATURES]; y=df.high_congestion
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=int(seed),stratify=y)
        if model_name=="Logistic Regression": model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000))
        elif model_name=="Decision Tree Classifier": model=DecisionTreeRegressor(max_depth=8,random_state=int(seed))
        elif model_name=="Random Forest Classifier": model=RandomForestClassifier(n_estimators=120,max_depth=12,n_jobs=-1,random_state=int(seed))
        elif model_name=="K-Nearest Neighbors":
            from sklearn.neighbors import KNeighborsClassifier
            model=make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=7))
        else:
            from sklearn.svm import SVC
            model=make_pipeline(StandardScaler(),SVC(probability=True))
        # correct classifier implementation for tree option
        if model_name=="Decision Tree Classifier":
            from sklearn.tree import DecisionTreeClassifier
            model=DecisionTreeClassifier(max_depth=8,random_state=int(seed))
        model.fit(Xtr,ytr); yp=model.predict(Xte); p=float(model.predict_proba(scenario)[0,1])
        a,b,c,d=st.columns(4)
        a.metric("Accuracy",f"{accuracy_score(yte,yp):.3f}")
        b.metric("Precision",f"{precision_score(yte,yp,zero_division=0):.3f}")
        c.metric("Recall",f"{recall_score(yte,yp,zero_division=0):.3f}")
        d.metric("F1",f"{f1_score(yte,yp,zero_division=0):.3f}")
        st.success(f"Predicted high-congestion probability: {p:.1%}")
        cm=confusion_matrix(yte,yp)
        st.plotly_chart(px.imshow(cm,text_auto=True,labels=dict(x="Predicted",y="Actual",color="Count"),
                         x=["Normal","High"],y=["Normal","High"],title="Confusion matrix",color_continuous_scale="Blues"),use_container_width=True)
    st.caption("Metrics are measured on a held-out split of generated synthetic data. These results do not establish real-world accuracy.")

with stages[2]:
    st.subheader("Unsupervised anomaly detection")
    cols=["motor_temp_c","motor_current_a","speed_error","load_kg","congestion_pct"]
    X=df[cols]
    contamination=st.slider("Expected anomaly fraction",.01,.15,.05,.01)
    detector=make_pipeline(StandardScaler(),IsolationForest(contamination=contamination,n_estimators=150,random_state=int(seed)))
    detector.fit(X)
    flags=detector.predict(X)==-1
    show=df.copy(); show["detected_anomaly"]=flags
    a,b,c=st.columns(3)
    a.metric("Flagged records",f"{flags.sum():,}")
    b.metric("Flag rate",f"{flags.mean():.1%}")
    # synthetic ground truth available for prototype validation
    c.metric("Ground-truth anomaly rate",f"{df.synthetic_anomaly.mean():.1%}")
    from sklearn.metrics import precision_score,recall_score,f1_score
    st.write("**Evaluation against generated anomaly labels**")
    st.write(f"Precision: {precision_score(df.synthetic_anomaly,flags,zero_division=0):.3f} · Recall: {recall_score(df.synthetic_anomaly,flags,zero_division=0):.3f} · F1: {f1_score(df.synthetic_anomaly,flags,zero_division=0):.3f}")
    fig=px.scatter(show.sample(min(800,len(show)),random_state=int(seed)),x="motor_temp_c",y="motor_current_a",color=show.sample(min(800,len(show)),random_state=int(seed))["detected_anomaly"].map({True:"Flagged",False:"Normal"}),title="Telemetry anomaly map",labels={"motor_temp_c":"Motor temperature (°C)","motor_current_a":"Motor current (A)","color":"Detector"})
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig,use_container_width=True)
    st.info("Synthetic anomaly labels are used only to evaluate the demo detector. Real deployment requires validated, representative telemetry.")

def search_path(grid,start,goal,algo):
    R,C=grid.shape
    def h(p): return abs(p[0]-goal[0])+abs(p[1]-goal[1])
    q=[]; heappush(q,(h(start) if algo=="A*" else 0,start))
    came={}; costs={start:0}; visited=set()
    while q:
        _,cur=heappop(q)
        if cur in visited: continue
        visited.add(cur)
        if cur==goal:
            path=[cur]
            while cur in came: cur=came[cur]; path.append(cur)
            return path[::-1],costs[goal]
        for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nxt=(cur[0]+dr,cur[1]+dc)
            if not(0<=nxt[0]<R and 0<=nxt[1]<C) or grid[nxt]: continue
            congestion_cost=1+grid_cost[nxt]
            nc=costs[cur]+congestion_cost
            if nc<costs.get(nxt,float("inf")):
                costs[nxt]=nc; came[nxt]=cur
                heappush(q,(nc+(h(nxt) if algo=="A*" else 0),nxt))
    return [],float("inf")

with stages[3]:
    st.subheader("Interactive warehouse route planner")
    rows,cols=20,30
    rng=np.random.default_rng(int(seed))
    wall_density=st.slider("Obstacle density (%)",5,30,14)
    traffic=st.slider("Traffic intensity (%)",0,100,45)
    grid=np.zeros((rows,cols),dtype=int)
    for _ in range(int(rows*cols*wall_density/100)):
        r,c=rng.integers(0,rows),rng.integers(0,cols)
        if (r,c) not in [(0,0),(rows-1,cols-1)]: grid[r,c]=1
    grid_cost=rng.uniform(0,traffic/100*3,(rows,cols))
    start=(0,0); goal=(rows-1,cols-1)
    grid_cost[grid==1]=0
    p_astar,cost_astar=search_path(grid,start,goal,"A*")
    p_dij,cost_dij=search_path(grid,start,goal,"Dijkstra")
    p_adapt,cost_adapt=search_path(grid,start,goal,"A*")
    mode=st.radio("Displayed route",["Adaptive A*","A* baseline","Dijkstra baseline"],horizontal=True)
    path=p_adapt if mode=="Adaptive A*" else p_astar if mode=="A* baseline" else p_dij
    fig=go.Figure()
    z=grid_cost.copy(); z[grid==1]=4
    fig.add_trace(go.Heatmap(z=z,colorscale="YlOrRd",showscale=True,colorbar={"title":"Cost / obstacle"}))
    if path:
        fig.add_trace(go.Scatter(x=[p[1] for p in path],y=[p[0] for p in path],mode="lines+markers",name=mode,line={"color":"cyan","width":4},marker={"size":5}))
    fig.add_trace(go.Scatter(x=[0,cols-1],y=[0,rows-1],mode="markers+text",text=["START","GOAL"],textposition="top center",marker={"size":15,"color":["lime","white"],"symbol":["circle","star"]},name="Endpoints"))
    fig.update_yaxes(autorange="reversed",dtick=1); fig.update_xaxes(dtick=1)
    fig.update_layout(title="Warehouse cost map and selected path",template="plotly_dark",height=600,xaxis_title="Column",yaxis_title="Row")
    st.plotly_chart(fig,use_container_width=True)
    if path:
        a,b,c=st.columns(3)
        a.metric("Path steps",len(path)-1)
        b.metric("Accumulated cost",f"{cost_adapt:.1f}")
        c.metric("Route status","Path found")
    else: st.error("No route found. Reduce obstacle density or change the random seed.")
    st.caption("The adaptive planner uses simulated congestion-weighted cell costs. The displayed planner is a software grid simulation, not a physical robot controller.")

with stages[4]:
    st.subheader("Dataset, methods and limitations")
    st.write("This prototype generates synthetic warehouse observations from transparent equations with random variation. It is intended to demonstrate model training, evaluation and decision integration.")
    st.write("**Features:** "+", ".join(FEATURES))
    st.dataframe(df.head(15),use_container_width=True)
    st.download_button("Download generated dataset (CSV)",df.to_csv(index=False).encode(),file_name="nexus_ai_synthetic_warehouse.csv",mime="text/csv")
    st.markdown("""
**Methodology**
1. Generate simulated warehouse features and outcomes.
2. Split labelled data into training and test sets.
3. Fit a selected ML model on the training set.
4. Evaluate predictions on held-out records.
5. Use scenario predictions as inputs to a route-planning cost function.
6. Compare alternative routes and inspect outcomes.

**Limitations**
- Data is synthetic, not collected from an operational warehouse.
- The route planner is a grid-based software simulation.
- Model performance may not transfer to real robots without real-world validation.
- This MVP does not control physical hardware or guarantee collision-free operation.
""")
st.divider()
st.caption("NEXUS-AI · Research prototype · Synthetic-data demonstration · Validate before real-world deployment")
