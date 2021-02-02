import numpy as np
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import json

# From local module henon_wrap.py
from henon_wrap import henon_wrap_2d as f

##### STYLE DEFAULTS #####

# the style arguments for the sidebar.
SIDEBAR_STYLE = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '20%',
    'padding': '20px 10px',
    'background-color': '#f8f9fa'
}

# the style arguments for the main content page.
CONTENT_STYLE = {
    'margin-left': '21%',
    'margin-right': '1%',
    'top': 0,
    'padding': '3px 3px'
}

TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#191970'
}

CARD_TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#0074D9'
}

# Heatmap dimention in pixels
heatmap_width = 700
bin_fraction = 100

##### SOME INITIAL VALUES #####
fineness = 250
n_turns_0 = 100
base_extent = [0,1,0,1]

##### DASH Framework #####
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

##### A BIT OF SETUP #####
indicators = f.get_indicators_available()

dropdown_options = [{'label':indicators[i], 'value':indicators[i]} for i in range(len(indicators))]
params_names = f.get_param_names()
params_defaults = f.get_param_defaults()


def recompute_data(**k):
    """We will call this function for recomputing the data for the heatmap!"""
    data = f.compute_indicator(k["extents"], k["n_turns"], k["sampling"],
                               *k["params"], method=k["method"], indicator=k["indicator"])
    return data.reshape((k["sampling"], k["sampling"]))

# Make a first instance of an heatmap, properly configured.
data = recompute_data(
    extents=[0,1,0,1,0,0],
    n_turns=100,
    sampling=fineness,
    params=f.get_param_defaults(),
    method="polar",
    indicator=f.get_indicators_available()[0]
)

fig = {
    'data':[{
    'z': data,
    'x': np.linspace(base_extent[0], base_extent[1], fineness),
    'y': np.linspace(base_extent[2], base_extent[3], fineness),
    'hoverongaps': False,
    'type': 'heatmap',
    }],
    'layout':
        {
            'width': heatmap_width,
            'height': heatmap_width
        }
}

fig_h = go.Figure(
    data=[go.Histogram(x=data.flatten(), histnorm='probability')]
)

##### DASHBOARD SETUP #####

controls = dbc.FormGroup(
    [
        html.P('Dynamic Indicator', style={
            'textAlign': 'center'
        }),
        dcc.Dropdown(
            id='dyn_indicator',
            options=dropdown_options,
            value=dropdown_options[0]["value"],
            multi=False
        ),
        html.Br(),
        html.P('Theta1', style={
            'textAlign': 'center'
        }),
        dcc.Slider(
            id='theta_1',
            min=0,
            max=2,
            step=0.01,
            value=0,
            marks={
                0: {'label': '0π'},
                0.5: {'label': '0.5π'},
                1: {'label': '1π'},
                1.5: {'label': '1.5π'},
                2: {'label': '2π'}
            },
            updatemode="drag"
        ),
        html.P('Theta2', style={
            'textAlign': 'center'
        }),
        dcc.Slider(
            id='theta_2',
            min=0,
            max=2,
            step=0.01,
            value=0,
            marks={
                0: {'label': '0π'},
                0.5: {'label': '0.5π'},
                1: {'label': '1π'},
                1.5: {'label': '1.5π'},
                2: {'label': '2π'}
            },
            updatemode="drag"
        ), 
        html.P('View system', style={
            'textAlign': 'center'
        }),
        dbc.Card([dbc.RadioItems(
            id='view_system',
            options=[{
                'label': 'Polar (Alpha=0.0)',
                'value': 'polar'
            },
                {
                    'label': 'X - PX',
                    'value': 'x_px'
            },
                {
                    'label': 'Y - PY',
                    'value': 'y_py'
            }
            ],
            value='polar',
            style={
                'margin': 'auto'
            }
        )]),
        html.P('Options', style={
            'textAlign': 'center'
        }),
        dbc.Card([dbc.Checklist(
            id='scale',
            options=[{
                'label': 'Log Scale (L)',
                'value': 'log_scale_l'
            }, {
                'label': 'Log Scale (R)',
                'value': 'log_scale_r'
            },
                {
                'label': 'Update theta on drag',
                'value': 'drag'
            }
            ],
            value=['log_scale_l', 'drag'],
            inline=True
        )]),
        html.Br(),
        html.P('Parameters', style={
            'textAlign': 'center'
        }),
        
        html.Div([
            html.Div("Samples per side:"),
            html.Div(
                dcc.Input(
                    id="fineness",
                    type="number",
                    placeholder="fineness",
                    value=fineness
                )
            ),
        ]),

        html.Div([
            html.Div("N turns (left):"),
            html.Div(
                dcc.Input(
                    id="nturns_l",
                    type="number",
                    placeholder="n turns for tracking",
                    value=n_turns_0
                )
            ),
        ]),

        html.Div([
            html.Div("N turns (right):"),
            html.Div(
                dcc.Input(
                    id="nturns_r",
                    type="number",
                    placeholder="n turns for indicator",
                    value=n_turns_0
                )
            ),
        ]),
    ] + [
        html.Div([
            html.Div(params_names[i]),
            html.Div(
                dcc.Input(
                    id="param_{}".format(i),
                    type="number",
                    placeholder=params_names[i],
                    value=params_defaults[i] 
                )
            ),
        ]) for i in range(len(params_names))
    ]
)

sidebar = html.Div(
    [
        html.H2('Dashboard', style=TEXT_STYLE),
        html.Hr(),
        controls
    ],
    style=SIDEBAR_STYLE,
)

##### HEATMAP ZONE #####

content_first_row = dbc.Row(
    [dbc.Col(
        dbc.Card(
        [dbc.CardBody(
            [
                html.H4(id='card_title_1', children=['Card Title 1'], className='card-title',
                        style=CARD_TEXT_STYLE),
                html.P(id='card_text_1', children=[
                    'Sample text.'], style=CARD_TEXT_STYLE),
            ]
        )]), md=6),
    dbc.Col(
        dbc.Card(
            [dbc.CardBody(
                [
                    html.H4(id='card_title_2', children=['Card Title 1'], className='card-title',
                            style=CARD_TEXT_STYLE),
                    html.P(id='card_text_2', children=[
                        'Sample text.'], style=CARD_TEXT_STYLE),
                ]
            )]), md=6)
    ])


content_second_row = dbc.Row(
    [
        dbc.Col(
            dcc.Graph(id='graph_1', figure=fig), md=6,
        ),
        dbc.Col(
            dcc.Graph(id='graph_2', figure=fig), md=6,
        )
    ]
)


content_third_row = dbc.Row(
    [
        dbc.Col(
            dcc.Graph(id='hist_1', figure=fig_h), md=6,
        ),
        dbc.Col(
            dcc.Graph(id='hist_2', figure=fig_h), md=6,
        )
    ]
)

content_fourt_row = dbc.Row(
    [
        dbc.Col(
            dcc.RangeSlider(
                id='range_slider_1',
                min=data.min(),
                max=data.max(),
                value=[data.min(), (data.min()+data.max())/3,
                       (data.min()+data.max())*2/3, data.max()],
                step=(data.max()-data.min()) / 100,
                allowCross=False
            ), md=6
        ),
        dbc.Col(
            html.Div(id='output_range_slider_1'), md=6
        )
    ]
)

content_fifth_row = dbc.Row(
    [
        dbc.Col(
            dcc.Graph(id='hist_3', figure=fig_h), md=6,
        ),
        dbc.Col(
            dcc.Graph(id='hist_4', figure=fig_h), md=6,
        )
    ]
)

content_sixth_row = dbc.Row(
    [
        dbc.Col(
            dcc.RangeSlider(
                id='range_slider_2',
                min=data.min(),
                max=data.max(),
                value=[data.min(), (data.min()+data.max())/3,
                       (data.min()+data.max())*2/3, data.max()],
                step=(data.max()-data.min()) / 100,
                allowCross=False
            ), md=6
        ),
        dbc.Col(
            html.Div(id='output_range_slider_2'), md=6
        )
    ][::-1]
)

content_seventh_row = dbc.Row(
    [
        dbc.Col(
            dcc.Graph(id='hist_5', figure=fig_h), md=6,
        ),
        dbc.Col(
            dcc.Graph(id='hist_6', figure=fig_h), md=6,
        )
    ]
)

content_eight_row = dbc.Row(
    [
        dbc.Col(
            dcc.Graph(id='scat_1', figure=fig_h), md=6,
        ),
        dbc.Col(
            dcc.Graph(id='scat_2', figure=fig_h), md=6,
        )
    ]
)


content = html.Div(
    [
        #html.H2('Analytics Dashboard Template'),
        #html.Hr(),
        content_first_row,
        content_second_row,
        content_third_row,
        content_fourt_row,
        content_fifth_row,
        content_sixth_row,
        content_seventh_row,
        content_eight_row,

        # Hidden div inside the app that stores the intermediate value
        html.Div(id='intermediate-value-1', style={'display': 'none'}),
        html.Div(id='intermediate-value-2', style={'display': 'none'}),
    ],
    style=CONTENT_STYLE
)

##### FINAL LAYOUT #####
app.layout = html.Div([sidebar, content])

##### CALLBACKS #####

@app.callback(Output('theta_1', 'updatemode'), Input('scale', 'value'))
def update_slider_th1(mode):
    """Switch for the update mode of th1 slider"""
    if "drag" in mode:
        return "drag"
    else:
        return "mouseup"


@app.callback(Output('theta_2', 'updatemode'), Input('scale', 'value'))
def update_slider_th2(mode):
    """Switch for the update mode of th2 slider"""
    if "drag" in mode:
        return "drag"
    else:
        return "mouseup"


@app.callback(
    Output('card_title_1', 'children'),
    [Input('scale', 'value')])
def update_card_text_1_title(*arg):
    """Callback for updating the card1 title"""
    return 'Tracking' + (" (log10 scale)" if "log_scale_l" in arg[0] else " (linear scale)")


@app.callback(
    Output('card_title_2', 'children'),
    [Input('dyn_indicator', 'value'),
     Input('scale', 'value')])
def update_card_text_2_title(*arg):
    """Callback for updating the card1 title"""
    return 'Displayed Dynamic Indicator: "{}"'.format(arg[0]) + (" (log10 scale)" if "log_scale_r" in arg[1] else " (linear scale)")


@app.callback(
    Output('card_text_1', 'children'),
    [Input('theta_1', 'value'),
     Input('theta_2', 'value'),
     Input('view_system', 'value')])
def update_card_text_1_content(*arg):
    """Callback for updating the card1 subtext"""
    if arg[-1] == "x_px":
        return "View System: X - PX plane"
    if arg[-1] == "y_py":
        return "View System: Y - PY plane"
    else:
        return "View System: r_x - r_y plane with theta1 = {}π, theta2 = {}π".format(arg[0], arg[1])


@app.callback(
    Output('card_text_2', 'children'),
    [Input('theta_1', 'value'),
     Input('theta_2', 'value'),
     Input('view_system', 'value')])
def update_card_text_2_content(*arg):
    """Callback for updating the card1 subtext"""
    if arg[-1] == "x_px":
        return "View System: X - PX plane"
    if arg[-1] == "y_py":
        return "View System: Y - PY plane"
    else:
        return "View System: r_x - r_y plane with theta1 = {}π, theta2 = {}π".format(arg[0], arg[1])


@app.callback(
    Output('intermediate-value-2', 'children'),
    [Input('graph_1', 'relayoutData'),  # 0
     Input('graph_2', 'relayoutData'),  # 1
     Input('dyn_indicator', 'value'),  # 2
     Input('theta_1', 'value'),  # 3
     Input('theta_2', 'value'),  # 4
     Input('view_system', 'value'),  # 5
     Input('scale', 'value'),  # 6
     Input('fineness', 'value'),  # 7
     Input('nturns_r', 'value')  # 8
     ]+[  # 9:-1
        Input("param_{}".format(i), 'value') for i in range(len(params_names))
    ],
    [State('graph_2', 'figure')])  # -1
def update_graph_2(*par):
    """General callback for updating the heatmap whenever a parameter is changed or a zoom/panning is performed."""
    ctx = dash.callback_context
    if "graph_1.relayoutData" in ctx.triggered[0]["prop_id"]:
        if 'dragmode' in par[0]:
            return json.dumps(par[-1]["data"][0]["z"])
        if 'xaxis.range[0]' in par[0]:
            base_extent[:] = [
                par[0]['xaxis.range[0]'],
                par[0]['xaxis.range[1]'],
                base_extent[2],
                base_extent[3]]
        if 'yaxis.range[0]' in par[0]:
            base_extent[:] = [
                base_extent[0],
                base_extent[1],
                par[0]['yaxis.range[0]'],
                par[0]['yaxis.range[1]']]
    if "graph_2.relayoutData" in ctx.triggered[0]["prop_id"]:
        if 'dragmode' in par[1]:
            return json.dumps(par[-1]["data"][0]["z"])
        if 'xaxis.range[0]' in par[1]:
            base_extent[:] = [
                par[1]['xaxis.range[0]'],
                par[1]['xaxis.range[1]'],
                base_extent[2],
                base_extent[3]]
        if 'yaxis.range[0]' in par[1]:
            base_extent[:] = [
                base_extent[0],
                base_extent[1],
                par[1]['yaxis.range[0]'],
                par[1]['yaxis.range[1]']]
    extent = base_extent.copy()

    if par[5] == "polar":
        extent.append(par[3] * np.pi)
        extent.append(par[4] * np.pi)
    data = recompute_data(
        extents=extent,
        n_turns=par[8],
        sampling=par[7],
        params=par[9:-1],
        method=par[5],
        indicator=par[2]
    )

    data_list = data.tolist()
    return json.dumps(data_list)


@app.callback(
    Output('graph_2', 'figure'),
    [Input('intermediate-value-2', 'children')],
    [State('graph_1', 'relayoutData'),  # 1
     State('graph_2', 'relayoutData'),  # 2
     State('dyn_indicator', 'value'),  # 3
     State('theta_1', 'value'),  # 4
     State('theta_2', 'value'),  # 5
     State('view_system', 'value'),  # 6
     State('scale', 'value'),  # 7
     State('fineness', 'value'),  # 8
     State('nturns_r', 'value')  # 9
     ]+[  # 10:-1
        State("param_{}".format(i), 'value') for i in range(len(params_names))
    ] +
    [State('graph_2', 'figure')])  # -1
def update_heatmap_2(*par):
    data = json.loads(par[0])
    extent = base_extent.copy()

    if par[6] == "polar":
        extent.append(par[4] * np.pi)
        extent.append(par[5] * np.pi)
    if "log_scale_r" in par[7]:
        data = np.log10(data)
        data[np.isinf(data)] = np.nan
    extra_data = f.compute_coords(extent, par[8], par[6])
    sig_digitx = int(
        np.ceil(-np.log10((extent[1] - extent[0]) / (par[8] * 2))))
    sig_digity = int(
        np.ceil(-np.log10((extent[3] - extent[2]) / (par[8] * 2))))

    fig = {
        'data': [{
            'z': data,
            'x': np.linspace(base_extent[0], base_extent[1], par[8]),
            'y': np.linspace(base_extent[2], base_extent[3], par[8]),
            'hoverongaps': False,
            'type': 'heatmap',
            'customdata': np.dstack((
                extra_data[0].reshape((par[8], par[8])),
                extra_data[1].reshape((par[8], par[8])),
                extra_data[2].reshape((par[8], par[8])),
                extra_data[3].reshape((par[8], par[8])),
            )),
            'hovertemplate': "<br>".join([
                "X0: %{customdata[0]:." + str(sig_digitx) + "f}",
                "PX0: %{customdata[1]:." + str(sig_digitx) + "f}",
                "Y0: %{customdata[2]:." + str(sig_digity) + "f}",
                "PY0: %{customdata[3]:." + str(sig_digity) + "f}",
                "Value: %{z}"
            ])
        }],
        'layout':
            {
                'width': heatmap_width,
                'height': heatmap_width,
        }

    }
    return fig


@app.callback(
    Output('hist_2', 'figure'),
    [Input('intermediate-value-2', 'children')],
    [State('graph_1', 'relayoutData'),  # 1
     State('graph_2', 'relayoutData'),  # 2
     State('dyn_indicator', 'value'),  # 3
     State('theta_1', 'value'),  # 4
     State('theta_2', 'value'),  # 5
     State('view_system', 'value'),  # 6
     State('scale', 'value'),  # 7
     State('fineness', 'value'),  # 8
     State('nturns_r', 'value')  # 9
     ]+[  # 10:-1
        State("param_{}".format(i), 'value') for i in range(len(params_names))
    ] +
    [State('graph_2', 'figure')])  # -1
def update_hist_2(*par):
    data = np.array(json.loads(par[0]))

    if "log_scale_r" in par[7]:
        data = np.log10(data)
        data[np.isinf(data)] = np.nan

    fig_h = go.Figure(
        data=[go.Histogram(x=data.flatten(), histnorm='probability')]
    )
    return fig_h


@app.callback(
    Output('intermediate-value-1', 'children'),
    [Input('graph_1', 'relayoutData'),  # 0
     Input('graph_2', 'relayoutData'),  # 1
     Input('theta_1', 'value'),  # 2
     Input('theta_2', 'value'),  # 3
     Input('view_system', 'value'),  # 4
     Input('fineness', 'value'),  # 5
     Input('nturns_l', 'value')  # 6
     ]+[  # 7:-1
        Input("param_{}".format(i), 'value') for i in range(len(params_names))
    ],
    [State('graph_1', 'figure')])  # -1
def update_graph_1(*par):
    """General callback for updating the heatmap whenever a parameter is changed or a zoom/panning is performed."""
    ctx = dash.callback_context
    if "graph_1.relayoutData" in ctx.triggered[0]["prop_id"]:
        if 'dragmode' in par[0]:
            return json.dumps(par[-1]["data"][0]["z"])
        if 'xaxis.range[0]' in par[0]:
            base_extent[:] = [
                par[0]['xaxis.range[0]'],
                par[0]['xaxis.range[1]'],
                base_extent[2],
                base_extent[3]]
        if 'yaxis.range[0]' in par[0]:
            base_extent[:] = [
                base_extent[0],
                base_extent[1],
                par[0]['yaxis.range[0]'],
                par[0]['yaxis.range[1]']]
    elif "graph_2.relayoutData" in ctx.triggered[0]["prop_id"]:
        if 'dragmode' in par[1]:
            return json.dumps(par[-1]["data"][0]["z"])
        if 'xaxis.range[0]' in par[1]:
            base_extent[:] = [
                par[1]['xaxis.range[0]'],
                par[1]['xaxis.range[1]'],
                base_extent[2],
                base_extent[3]]
        if 'yaxis.range[0]' in par[1]:
            base_extent[:] = [
                base_extent[0],
                base_extent[1],
                par[1]['yaxis.range[0]'],
                par[1]['yaxis.range[1]']]
    extent = base_extent.copy()

    if par[4] == "polar":
        extent.append(par[2] * np.pi)
        extent.append(par[3] * np.pi)
    data = recompute_data(
        extents=extent,
        n_turns=par[6],
        sampling=par[5],
        params=par[7:-1],
        method=par[4],
        indicator="tracking"
    )

    data_list = data.tolist()
    return json.dumps(data_list)


@app.callback(
    Output('graph_1', 'figure'),
    [Input('intermediate-value-1', 'children')],
    [State('graph_1', 'relayoutData'),  # 1
     State('graph_2', 'relayoutData'),  # 2
     State('theta_1', 'value'),  # 3
     State('theta_2', 'value'),  # 4
     State('view_system', 'value'),  # 5
     State('scale', 'value'),  # 6
     State('fineness', 'value'),  # 7
     State('nturns_r', 'value')  # 8
     ]+[  # 9:-1
        State("param_{}".format(i), 'value') for i in range(len(params_names))
    ] +
    [State('graph_1', 'figure')])  # -1
def update_heatmap_1(*par):
    data = json.loads(par[0])
    extent = base_extent.copy()

    if par[5] == "polar":
        extent.append(par[3] * np.pi)
        extent.append(par[4] * np.pi)
    if "log_scale_l" in par[6]:
        data = np.log10(data)
        data[np.isinf(data)] = np.nan
    extra_data = f.compute_coords(extent, par[7], par[5])
    sig_digitx = int(
        np.ceil(-np.log10((extent[1] - extent[0]) / (par[7] * 2))))
    sig_digity = int(
        np.ceil(-np.log10((extent[3] - extent[2]) / (par[7] * 2))))

    fig = {
        'data': [{
            'z': data,
            'x': np.linspace(base_extent[0], base_extent[1], par[7]),
            'y': np.linspace(base_extent[2], base_extent[3], par[7]),
            'hoverongaps': False,
            'type': 'heatmap',
            'customdata': np.dstack((
                extra_data[0].reshape((par[7], par[7])),
                extra_data[1].reshape((par[7], par[7])),
                extra_data[2].reshape((par[7], par[7])),
                extra_data[3].reshape((par[7], par[7])),
            )),
            'hovertemplate': "<br>".join([
                "X0: %{customdata[0]:." + str(sig_digitx) + "f}",
                "PX0: %{customdata[1]:." + str(sig_digitx) + "f}",
                "Y0: %{customdata[2]:." + str(sig_digity) + "f}",
                "PY0: %{customdata[3]:." + str(sig_digity) + "f}",
                "Value: %{z}"
            ])
        }],
        'layout':
            {
                'width': heatmap_width,
                'height': heatmap_width,
        }

    }
    return fig


@app.callback(
    Output('hist_1', 'figure'),
    [Input('intermediate-value-1', 'children')],
    [State('graph_1', 'relayoutData'),  # 1
     State('graph_2', 'relayoutData'),  # 2
     State('dyn_indicator', 'value'),  # 3
     State('theta_1', 'value'),  # 4
     State('theta_2', 'value'),  # 5
     State('view_system', 'value'),  # 6
     State('scale', 'value'),  # 7
     State('fineness', 'value'),  # 8
     State('nturns_r', 'value')  # 9
     ]+[  # 10:-1
        State("param_{}".format(i), 'value') for i in range(len(params_names))
    ] +
    [State('graph_1', 'figure')])  # -1
def update_hist_1(*par):
    data = np.array(json.loads(par[0]))

    if "log_scale_l" in par[7]:
        data = np.log10(data)
        data[np.isinf(data)] = np.nan

    fig_h = go.Figure(
        data=[go.Histogram(x=data.flatten(), histnorm='probability')]
    )
    return fig_h


@app.callback(
    Output('range_slider_1', 'max'),
    [Input('intermediate-value-1', 'children')],
    [State('scale', 'value')]
)
def update_r_slider_max(*par):
    data = np.array(json.loads(par[0]))
    if "log_scale_l" in par[1]:
        data = np.log10(data)
        data[np.isinf(data)] = np.nan
    #print(np.nanmax(data))
    return np.nanmax(data)


@app.callback(
    Output('range_slider_1', 'min'),
    [Input('intermediate-value-1', 'children')],
    [State('scale', 'value')]
)
def update_r_slider_min(*par):
    data = np.array(json.loads(par[0]))
    if "log_scale_l" in par[1]:
        data = np.log10(data)
        data[np.isinf(data)] = np.nan
    #print(np.nanmin(data))
    return np.nanmin(data)


@app.callback(
    Output('range_slider_1', 'value'),
    [Input('range_slider_1', 'min'),
     Input('range_slider_1', 'max')]
)
def update_r_slider_value(*par):
    #print([par[0], (par[0] + par[1]) / 2, par[1]])
    return [par[0], (par[0] + par[1]) / 3, (par[0] + par[1]) * 2 / 3, par[1]]


@app.callback(
    Output('range_slider_1', 'step'),
    [Input('range_slider_1', 'min'),
     Input('range_slider_1', 'max')]
)
def update_r_slider_step(*par):
    return (par[1] - par[0]) / 100


@app.callback(
    Output('output_range_slider_1', 'children'),
    Input('range_slider_1', 'value')
)
def update_r_slider_print(values):
    return "Stability bool masks. 1st range: [{:.4f},{:.4f}]. 2nd range: [{:.4f},{:.4f}]".format(values[0], values[1], values[2], values[3])


@app.callback(
    Output('hist_3', 'figure'),
    Input('range_slider_1', 'value'),
    [
        State('intermediate-value-1', 'children'),
        State('scale', 'value')
    ]
)
def update_slider_hist_left(*par):
    data = np.array(json.loads(par[1]))
    if "log_scale_l" in par[2]:
        data = np.log10(data)
        data[np.isinf(data)] = np.nan
    data1 = data[np.logical_and(data >= par[0][0], data <= par[0][1])]
    data2 = data[np.logical_and(data >= par[0][2], data <= par[0][3])]

    start = np.nanmin(data) * 0.9999
    end = np.nanmax(data) * 1.0001
    size = (end - start) / bin_fraction

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=data1,
        xbins=dict(
            start=start,
            end=end,
            size=size),
        autobinx=False)
    )
    fig.add_trace(go.Histogram(
        x=data2,
        xbins=dict(
            start=start,
            end=end,
            size=size),
        autobinx=False)
    )
    # Overlay both histograms
    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.75)

    fig.update_layout(
        title="Stability histogram (selected intervals)",
        xaxis_title="Stability value" +
        (" [linear]" if "log_scale_l" in par[2] else " [log10]"),
        yaxis_title="Occurrences",
        legend_title="Intervals",
    )
    return fig


@app.callback(
    Output('hist_4', 'figure'),
    Input('range_slider_1', 'value'),
    [
        State('intermediate-value-1', 'children'),
        State('intermediate-value-2', 'children'),
        State('scale', 'value')
    ]
)
def update_slider_hist_right(*par):
    data_l = np.array(json.loads(par[1])).flatten()
    if "log_scale_l" in par[3]:
        data_l = np.log10(data_l)
        data_l[np.isinf(data_l)] = np.nan
    data_r = np.array(json.loads(par[2])).flatten()
    if "log_scale_r" in par[3]:
        data_r = np.log10(data_r)
        data_r[np.isinf(data_r)] = np.nan
    data1 = data_r[np.logical_and(data_l >= par[0][0], data_l <= par[0][1])]
    data2 = data_r[np.logical_and(data_l >= par[0][2], data_l <= par[0][3])]

    start=np.nanmin(data_r) * 0.9999
    end=np.nanmax(data_r) * 1.0001
    size = (end-start)/bin_fraction

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=data1,
        xbins=dict(
            start=start,
            end=end,
            size=size),
        autobinx=False)
    )
    fig.add_trace(go.Histogram(
        x=data2,
        xbins=dict(
            start=start,
            end=end,
            size=size),
        autobinx=False)
    )

    # Overlay both histograms
    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.75)
    fig.update_layout(
        title="Dynamic indicator (ranges selected on the left)",
        xaxis_title="D.I. value" +
        (" [linear]" if "log_scale_l" in par[2] else " [log10]"),
        yaxis_title="Occurrences",
        legend_title="Intervals",
    )
    return fig


@app.callback(
    Output('range_slider_2', 'max'),
    [Input('intermediate-value-2', 'children')],
    [State('scale', 'value')]
)
def update_r_slider_max_2(*par):
    data = np.array(json.loads(par[0]))
    if "log_scale_r" in par[1]:
        data = np.log10(data)
        data[np.isinf(data)] = np.nan
    #print(np.nanmax(data))
    return np.nanmax(data)


@app.callback(
    Output('range_slider_2', 'min'),
    [Input('intermediate-value-2', 'children')],
    [State('scale', 'value')]
)
def update_r_slider_min_2(*par):
    data = np.array(json.loads(par[0]))
    if "log_scale_r" in par[1]:
        data = np.log10(data)
        data[np.isinf(data)] = np.nan
    #print(np.nanmin(data))
    return np.nanmin(data)


@app.callback(
    Output('range_slider_2', 'value'),
    [Input('range_slider_2', 'min'),
     Input('range_slider_2', 'max')]
)
def update_r_slider_value_2(*par):
    #print([par[0], (par[0] + par[1]) / 2, par[1]])
    return [par[0], (par[0] + par[1]) / 3, (par[0] + par[1]) * 2 / 3, par[1]]


@app.callback(
    Output('range_slider_2', 'step'),
    [Input('range_slider_2', 'min'),
     Input('range_slider_2', 'max')]
)
def update_r_slider_step_2(*par):
    return (par[1] - par[0]) / 100


@app.callback(
    Output('output_range_slider_2', 'children'),
    Input('range_slider_2', 'value')
)
def update_r_slider_print_2(values):
    return "Dyn Indicator bool masks. 1st range: [{:.4f},{:.4f}]. 2nd range: [{:.4f},{:.4f}]".format(values[0], values[1], values[2], values[3])


@app.callback(
    Output('hist_6', 'figure'),
    Input('range_slider_2', 'value'),
    [
        State('intermediate-value-2', 'children'),
        State('scale', 'value')
    ]
)
def update_slider_hist_right_2(*par):
    data = np.array(json.loads(par[1]))
    if "log_scale_r" in par[2]:
        data = np.log10(data)
        data[np.isinf(data)] = np.nan
    data1 = data[np.logical_and(data >= par[0][0], data < par[0][1])]
    data2 = data[np.logical_and(data >= par[0][2], data <= par[0][3])]

    start = np.nanmin(data) * 0.9999
    end = np.nanmax(data) * 1.0001
    size = (end - start) / bin_fraction

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=data1,
        xbins=dict(
            start=start,
            end=end,
            size=size),
        autobinx=False)
    )
    fig.add_trace(go.Histogram(
        x=data2,
        xbins=dict(
            start=start,
            end=end,
            size=size),
        autobinx=False)
    )

    # Overlay both histograms
    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.75)
    fig.update_layout(
        title="Dynamic Indicator (selected intervals)",
        xaxis_title="D.I. value" +
        (" [linear]" if "log_scale_l" in par[2] else " [log10]"),
        yaxis_title="Occurrences",
        legend_title="Intervals",
    )
    return fig


@app.callback(
    Output('hist_5', 'figure'),
    Input('range_slider_2', 'value'),
    [
        State('intermediate-value-1', 'children'),
        State('intermediate-value-2', 'children'),
        State('scale', 'value')
    ]
)
def update_slider_hist_left_2(*par):
    data_l = np.array(json.loads(par[1])).flatten()
    if "log_scale_l" in par[3]:
        data_l = np.log10(data_l)
        data_l[np.isinf(data_l)] = np.nan
    data_r = np.array(json.loads(par[2])).flatten()
    if "log_scale_r" in par[3]:
        data_r = np.log10(data_r)
        data_r[np.isinf(data_r)] = np.nan
    data1 = data_l[np.logical_and(data_r >= par[0][0], data_r < par[0][1])]
    data2 = data_l[np.logical_and(data_r >= par[0][2], data_r <= par[0][3])]

    start = np.nanmin(data_l) * 0.9999
    end = np.nanmax(data_l) * 1.0001
    size = (end - start) / bin_fraction

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=data1,
        xbins=dict(
            start=start,
            end=end,
            size=size),
        autobinx=False)
    )
    fig.add_trace(go.Histogram(
        x=data2,
        xbins=dict(
            start=start,
            end=end,
            size=size),
        autobinx=False)
    )
    # Overlay both histograms
    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.75)
    fig.update_layout(
        title="Stability histogram (intervals selected on right)",
        xaxis_title="Stability value" +
        (" [linear]" if "log_scale_l" in par[2] else " [log10]"),
        yaxis_title="Occurrences",
        legend_title="Intervals",
    )
    return fig


@app.callback(
    Output('scat_1', 'figure'),
    [Input('intermediate-value-1', 'children'),
     Input('intermediate-value-2', 'children')],
    State('scale', 'value')
)
def update_scatter_1(*par):
    data1 = np.array(json.loads(par[0]))
    if "log_scale_l" in par[2]:
        data1 = np.log10(data1)
        data1[np.isinf(data1)] = np.nan
    data2 = np.array(json.loads(par[1]))
    if "log_scale_r" in par[2]:
        data2 = np.log10(data2)
        data2[np.isinf(data2)] = np.nan
    fig = go.Figure(data=go.Scattergl(x=data1.flatten(),
                                      y=data2.flatten(), mode='markers'))
    fig.update_layout(width=800, height=800)
    fig.update_layout(
        title="Scatter map",
        xaxis_title="Tracking value" + (" [log10]" if "log_scale_l" in par[2] else " [linear]"),
        yaxis_title="Dynamic indicator" +
        (" [log10]" if "log_scale_r" in par[2] else " [linear]"),
        legend_title="Legend Title",
    )
    return fig


@app.callback(
    Output('scat_2', 'figure'),
    [Input('intermediate-value-1', 'children'),
     Input('intermediate-value-2', 'children')],
    State('scale', 'value')
)
def update_scatter_2(*par):
    data1 = np.array(json.loads(par[0]))
    if "log_scale_l" in par[2]:
        data1 = np.log10(data1)
        data1[np.isinf(data1)] = np.nan
    data2 = np.array(json.loads(par[1]))
    if "log_scale_r" in par[2]:
        data2 = np.log10(data2)
        data2[np.isinf(data2)] = np.nan

    startx = np.nanmin(data1[np.logical_not(np.isnan(data2))]) 
    endx = np.nanmax(data1[np.logical_not(np.isnan(data2))]) 
    sizex = (endx - startx) / 25
    starty = np.nanmin(data2[np.logical_not(np.isnan(data1))]) 
    endy = np.nanmax(data2[np.logical_not(np.isnan(data1))]) 
    sizey = (endy - starty) / 25

    fig = go.Figure(data=go.Histogram2d(
        x=data1.flatten(),
        y=data2.flatten(),
        #autobinx=False,
        #xbins=dict(start=startx, end=endx, size=sizex),
        #autobiny=False,
        #ybins=dict(start=starty, end=endy, size=sizey),
        colorscale=[
            [0, 'rgb(250, 250, 250)'],  # 0
            [1./10000, 'rgb(200, 200, 200)'],  # 10
            [1./1000, 'rgb(150, 150, 150)'],  # 100
            [1./100, 'rgb(100, 100, 100)'],  # 1000
            [1./10, 'rgb(50, 50, 50)'],  # 10000
            [1., 'rgb(0, 0, 0)'],  # 100000

        ],
        colorbar=dict(
            tick0=0,
            tickmode='array',
            tickvals=[0, 10, 100, 1000]
        )
    ))
    fig.update_layout(width=800, height=800)
    fig.update_layout(
        title="Density heatmap",
        xaxis_title="Tracking value" +
        (" [log10]" if "log_scale_l" in par[2] else " [linear]"),
        yaxis_title="Dynamic indicator" +
        (" [log10]" if "log_scale_r" in par[2] else " [linear]"),
        legend_title="Legend Title",
    )
    return fig


##### RUN THE SERVER #####
app.run_server(debug=True)
##########################
