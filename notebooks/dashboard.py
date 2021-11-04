import dash
from dash import dcc
from dash import html
from dash.dash_table.DataTable import DataTable
from dash.dependencies import Input, Output
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import dash_daq as daq
from dashboard_utils import get_elo_df, get_profit_df
import dash_table as dt
import numpy as np

df = pd.read_csv('../inputs/ready_data/preprocessed_all_matches.csv')
df = df[['HomeTeam', 'AwayTeam', 'Full_Time_Result', 'Season']]

elo_df = get_elo_df()
profit_df = get_profit_df()

df_home = df.groupby(['HomeTeam', 'Full_Time_Result', 'Season']).size().reset_index()
df_home.rename(columns = {0: 'Count'}, inplace=True)

df_away = df.groupby(['AwayTeam', 'Full_Time_Result', 'Season']).size().reset_index()
df_away.rename(columns = {0: 'Count'}, inplace=True)

all_dims = ['HomeTeam', 'Season']

drop_seasons = df.Season.unique().tolist()
drop_seasons.append('All')
teams = df.HomeTeam.unique()
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Lo del furboh'),
    dcc.Tabs(id="tabs-example-graph", value='tab-1-example-graph', children=[
        dcc.Tab(label='Main Stats', value='Main Stats'),
        dcc.Tab(label='Profit', value='Profit Plots'),

    ]),
    dcc.Dropdown(id="dropdown",options=[{"label": x, "value": x} for x in drop_seasons],value=all_dims[:2],
        multi=False
    ),
    html.Div(id='tabs-content-example-graph')
])

@app.callback(Output('tabs-content-example-graph', 'children'),
              Input('tabs-example-graph', 'value'))
def render_content(tab):
    if tab == 'Main Stats':
        return html.Div(children=[
        dbc.Row(children=[
            dbc.Col(children=[
                dcc.Dropdown( id="dropdown_home", options=[{"label": x, "value": x}  for x in teams], # value=all_dims[:2],
                            multi=False),
                        dcc.Graph(id="HomeTeam", style={'display': 'inline-block'})
                ], style={'display': 'inline-block'}
            ),
            dbc.Col(
                children=[
                    dcc.Dropdown(id="dropdown_away", options=[{"label": x, "value": x} for x in teams], # value=all_dims[:2],
                                multi=False),
                    dcc.Graph(id="AwayTeam", style={'display': 'inline-block'})
                ], style={'display': 'inline-block'}
            )
        ]),
        html.Div(children=[
            dcc.Graph(id="SeasonELO")
            ])
        ,
            html.Div(id="table1", style={"width": "90%", "height": "90%", "display": "inline-block", 
                                        "justify": "center", "align": "center"})
    ])
    elif tab == 'Profit Plots':
        return html.Div(children=[
            dcc.Graph(id="ProfitPlot"),
            dcc.Graph(id="ACCProfitPlot")
        ])



@app.callback(
    Output("HomeTeam", "figure"), 
    [Input("dropdown", "value"), Input("dropdown_home", "value")])
def update_bar_chart(season, team):
    if season == 'All':
        data_home = df_home.query('HomeTeam==@team')
    else:
        data_home = df_home.query('Season==@season & HomeTeam==@team')

    fig = px.bar(
        data_home, x='Full_Time_Result',y='Count', color="HomeTeam", barmode='group')
    return fig

@app.callback(
    Output("AwayTeam", "figure"), 
    [Input("dropdown", "value"), Input("dropdown_away", "value")])
def update_bar_chart(season, team):
    if season == 'All':
        data_away = df_away.query('AwayTeam==@team')
    else:
        data_away = df_away.query('Season==@season & AwayTeam==@team')
    fig = px.bar(
        data_away, x='Full_Time_Result',y='Count', color="AwayTeam", barmode='group',
        color_discrete_sequence =['red']*len(df))
    return fig


@app.callback(
    Output("SeasonELO", "figure"), 
    [Input("dropdown", "value"), Input("dropdown_home", "value"), Input("dropdown_away", "value")])
def update_lineplot(season, home, away):
    if season == 'All':
        data = elo_df.query('Team==@home | Team==@away')
    else:
        data = elo_df.query('Season==@season & (Team==@home | Team==@away)')
    fig = px.line(
        data, x='Date', y='ELO', color='Team'
    )
    return fig

@app.callback(
    Output("ProfitPlot", "figure"), 
    [Input("dropdown", "value")])
def update_lineplot(season):
    if season == 'All':
        data = profit_df
    else:
        data = profit_df.query('Season==@season')
    fig = px.line(
        data, x='Date', y='PROFIT', color='METHOD'
    )
    return fig

@app.callback(
    Output("ACCProfitPlot", "figure"), 
    [Input("dropdown", "value")])
def update_lineplot(season):
    if season == 'All':
        data = profit_df
    else:
        data = profit_df.query('Season==@season')
    fig = px.line(
        data, x='Date', y='ACC_PROFIT', color='METHOD'
    )
    return fig


@app.callback(
    dash.dependencies.Output('table1', 'children'),
    [Input("dropdown", "value"), Input("dropdown_home", "value"), Input("dropdown_away", "value")])
def update_output(season, home, away):
    if season == 'All':
        data = df.query('((HomeTeam==@home & AwayTeam==@away) | (HomeTeam==@away & AwayTeam==@home))')
    else:
        data = df.query('Season==@season & ((HomeTeam==@home & AwayTeam==@away) | (HomeTeam==@away & AwayTeam==@home))')
    data_1 = data.to_dict('rows')
    columns =  [{"name": i, "id": i,} for i in (df.columns)]
    return dt.DataTable(data=data_1, columns=columns)
    



app.run_server(debug=True)
