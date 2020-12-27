import numpy as np
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

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
    'margin-left': '25%',
    'margin-right': '5%',
    'top': 0,
    'padding': '20px 10px'
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
heatmap_width = 850

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
                'label': 'Log Scale',
                'value': 'log_scale'
            },
                {
                'label': 'Update theta on drag',
                'value': 'drag'
            }
            ],
            value=['log_scale', 'drag'],
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
            html.Div("N turns:"),
            html.Div(
                dcc.Input(
                    id="nturns",
                    type="number",
                    placeholder="nturns",
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

content_first_row = dbc.Card(
    [

        dbc.CardBody(
            [
                html.H4(id='card_title_1', children=['Card Title 1'], className='card-title',
                        style=CARD_TEXT_STYLE),
                html.P(id='card_text_1', children=[
                        'Sample text.'], style=CARD_TEXT_STYLE),
            ]
        )
    ]
)

content_second_row = dbc.Row(
    [
        dbc.Col(
            dcc.Graph(id='graph_1', figure=fig), md=12,
        )
    ]
)

content = html.Div(
    [
        #html.H2('Heatmap', style=TEXT_STYLE),
        #html.H2('Analytics Dashboard Template'),
        #html.Hr(),
        content_first_row,
        content_second_row
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
    [Input('dyn_indicator', 'value'),
     Input('scale', 'value')])
def update_card_text_1_title(*arg):
    """Callback for updating the card1 title"""
    return 'Displayed Dynamic Indicator: "{}"'.format(arg[0]) + " (log10 scale)" if "log_scale" in arg[1] else " (linear scale)"


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
    Output('graph_1', 'figure'),
    [Input('graph_1', 'relayoutData'),
     Input('dyn_indicator', 'value'),
     #Input('alpha', 'value'),
     Input('theta_1', 'value'),
     Input('theta_2', 'value'),
     Input('view_system', 'value'),
     Input('scale', 'value'),
     Input('fineness', 'value'),
     Input('nturns', 'value')
    ]+[
     Input("param_{}".format(i), 'value') for i in range(len(params_names))
    ],
    [State('graph_1', 'figure')])
def update_graph_1(*par):
    """General callback for updating the heatmap whenever a parameter is changed or a zoom/panning is performed."""
    if par[0] is not None: 
        if 'dragmode' in par[0]:
            return par[-1]
        if 'xaxis.range[0]' in par[0] and 'yaxis.range[0]' in par[0]:
            base_extent[:] = [
                par[0]['xaxis.range[0]'], par[0]['xaxis.range[1]'], par[0]['yaxis.range[0]'], par[0]['yaxis.range[1]']]
    extent = base_extent.copy()

    if par[4] == "polar":
        extent.append(par[2] * np.pi)
        extent.append(par[3] * np.pi)
    data = recompute_data(
        extents=extent,
        n_turns=par[7],
        sampling=par[6],
        params=par[8:-1],
        method=par[4],
        indicator=par[1]
    )

    if "log_scale" in par[5]:
        data = np.log10(data)
        data[np.isinf(data)] = -24.0

    extra_data = f.compute_coords(extent, par[6], par[4])
    
    sig_digitx = int(np.ceil(-np.log10((extent[1] - extent[0]) / (par[6] * 2))))
    sig_digity = int(np.ceil(-np.log10((extent[3] - extent[2]) / (par[6] * 2))))
    
    fig = {
        'data': [{
            'z': data,
            'x': np.linspace(base_extent[0], base_extent[1], par[6]),
            'y': np.linspace(base_extent[2], base_extent[3], par[6]),
            'hoverongaps': False,
            'type': 'heatmap',
            'customdata': np.dstack((
                extra_data[0].reshape((fineness, fineness)),
                extra_data[1].reshape((fineness, fineness)),
                extra_data[2].reshape((fineness, fineness)),
                extra_data[3].reshape((fineness, fineness)),
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

##### RUN THE SERVER #####
app.run_server(debug=True)
##########################
