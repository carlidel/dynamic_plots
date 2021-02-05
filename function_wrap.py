import numpy as np
import henon_map as hm


class function_wrap:
    """This class defines a standard wrapper for a generic 4D map function with an arbitrary number of dynamic indicators defined.
    """    
    def __init__(self, name, f, param_names=[], param_defaults=[]):
        """Initialize a standard function wrapper.

        Parameters
        ----------
        name : string
            name of the function
        f : function
            the base function defining the dynamic map, must have the following signature:
            f ((x, px, y, py), n_turns, start_point, *params)
            where:
            - (x, px, y, py) is a tuple of arrays of equal size containing the initial conditions to iterate.
            - n_turns is the number of iterations to perform
            - start_point is the number of iterations "already performed" (this can be an element that has it's own meaning when considering time-dependent modulations)
            - *params extra parameters which define the dynamic map's characteristics.
            And must return the resulting position in the form:
            (x, px, y, py)
        param_names : list of strings, optional
            a list containing the names of the map's extra parameters, by default []
        param_defaults : list of floats, optional
            a list containing the default values of the map's extra parameters, by default []
        """        
        assert len(param_names) == len(param_defaults)
        self.name = name
        self.f = f
        self.param_names = list(param_names)
        self.param_defaults = list(param_defaults)
        
        self.i_func = {}
        self.i_func_par_names = {}
        self.i_func_par_defaults = {}
        self.inv_f = None

    def __call__(self, init_list, n_turns, *params, start_point=0):
        """Call the base function of the dynamic map.

        Parameters
        ----------
        init_list : 4-element tuple of arrays
            initial conditions in the form of (x, px, y, py)
        n_turns : unsigned int
            number of iterations to perform
        start_point : int, optional
            number of iterations already done to be considered, by default 0
        *params : extra parameters of the function

        Returns
        -------
        tuple
            (x, px, y, py)
        """        
        return self.f(init_list, n_turns, start_point, *params)

    def compute_inverse(self, init_list, n_turns, *params, start_point=0):
        """Call the inverse function of the dynamic map.

        Parameters
        ----------
        init_list : 4-element tuple of arrays
            initial conditions in the form of (x, px, y, py)
        n_turns : unsigned int
            number of iterations to perform
        start_point : int, optional
            number of iterations already done to be considered, by default 0
        *params : extra parameters of the function

        Returns
        -------
        tuple
            (x, px, y, py)
        """
        assert self.inv_f is not None
        return self.inv_f(init_list, n_turns, start_point, *params)

    def compute_indicator(self, init_list, n_turns, *params, indicator_name=None, **kwargs):
        """Compute the required dynamic indicator.

        Parameters
        ----------
        init_list : tuple
            tuple of initial conditions (x, px, y, py)
        n_turns : unsigned int
            number of turns to perform
        indicator_name : string, optional
            name of the required dynamic indicator, by default None.
            If, none, will perform a call of the base function.

        Returns
        -------
        numpy array
            dynamic indicator values
        """        
        if indicator_name is None:
            self.__call__(init_list, n_turns, *params)
        assert indicator_name in self.i_func
        return self.i_func[indicator_name](init_list, n_turns, *params, **kwargs)

    def set_inverse(self, f):
        """Set the inverse function

        Parameters
        ----------
        f : function
            function defining the inverse dynamic map, must have the following signature:
            f ((x, px, y, py), n_turns, start_point, *params)
            where:
            - (x, px, y, py) is a tuple of arrays of equal size containing the initial conditions to iterate.
            - n_turns is the number of iterations to perform
            - start_point is the number of iterations "already performed" (this can be an element that has it's own meaning when considering time-dependent modulations)
            - *params extra parameters which define the dynamic map's characteristics.
            And must return the resulting position in the form:
            (x, px, y, py)
        """        
        self.inv_f = f

    def remove_inverse(self):
        """Removes the inverse function
        """        
        self.inv_f = None

    def set_indicator(self, f, name, ind_par_names=[], ind_par_defaults=[]):
        """Add a dynamic indicator with a defined name

        Parameters
        ----------
        f : function
            function defining the dynamic indicator, must have the following signature:
            f ((x, px, y, py), n_turns, *params)
            where:
            - (x, px, y, py) is a tuple of arrays of equal size containing the initial conditions to iterate.
            - n_turns is the number of iterations to perform
            - *params extra parameters which define the dynamic map's characteristics.
            And must return the resulting position in the form of a numpy array
        name : string
            the name of the indicator
        ind_par_names : list of strings
            name of internal parameters for the dynamic indicator
        ind_par_defaults : list of floats
            default values for dynamic indicator parameters
        """        
        self.i_func[name] = f
        self.i_func_par_names[name] = ind_par_names
        self.i_func_par_defaults[name] = ind_par_defaults

    def remove_indicator(self, name):
        """Remove the specified dynamic indicator

        Parameters
        ----------
        name : string
            name of the indicator to be removed
        """        
        if name in self.i_func:
            del self.i_func[name]
            del self.i_func_par_defaults[name]
            del self.i_func_par_names[name]

    def get_indicators_available(self):
        """Get the name list of the available dynamic indicators

        Returns
        -------
        list
            list of strings with the dynamic indicator names
        """        
        return list(self.i_func.keys())

    def get_indicator_param_names(self, name):
        return self.i_func_par_names[name]

    def get_indicator_param_defaults(self, name):
        return self.i_func_par_defaults[name]

    def get_param_names(self):
        """Get the names of the extra parameters 

        Returns
        -------
        list
            list of strings with extra parameter names
        """        
        return self.param_names

    def get_param_defaults(self):
        """Get the default values of the extra parameters

        Returns
        -------
        list
            list of floats
        """        
        return self.param_defaults


class function_multi:
    """Given a function wrapper instance, this class enhances it for easy usage in a 2D visualization environment.
    """    
    def __init__(self, f_wrapper):
        """Initialize a function_multi object.

        Parameters
        ----------
        f_wrapper : function_wrapper
            function wrapper object
        """        
        self.f = f_wrapper

    def __call__(self, extents, n_turns, sampling, *params, start_point=0, method="polar"):
        """Call the basic function with a 2D approach

        Parameters
        ----------
        extents : list
            list with the extents to use with the following protocol:
            - method == "polar" -> extent = [min_x, max_x, min_y, max_y, theta1, theta2]
            - method == "x_px" or "y_py" -> extent = [min_x, max_x, min_y, max_y]
        n_turns : unsigned int
            turns to perform
        sampling : unsigned int
            samplings to perform on each side
        start_point : int, optional
            number of already done turns to consider, by default 0
        method : str, optional
            choose between "polar", "x_px" or "y_py", by default "polar"
        *params : extra parameters for f

        Returns
        -------
        tuple
            tuple with all the computed values flattened (x, px, y, py)
        """        
        assert len(params) <= len(self.f.get_param_names())
        coords = self.compute_coords(extents, sampling, method)
        actual_params = list(params) + self.f.get_param_defaults()[len(params):len(self.f.get_param_defaults())]
        return self.f(coords, n_turns, start_point, *actual_params)

    def compute_indicator(self, extents, n_turns, sampling, *params, method="polar", indicator=None, **kwargs):
        """Call the required dynamic indicator with the given parameters

        Parameters
        ----------
        xtents : list
            list with the extents to use with the following protocol:
            - method == "polar" -> extent = [min_x, max_x, min_y, max_y, theta1, theta2]
            - method == "x_px" or "y_py" -> extent = [min_x, max_x, min_y, max_y]
        n_turns : unsigned int
            turns to perform
        sampling : unsigned int
            samplings to perform on each side
        method : str, optional
            choose between "polar", "x_px" or "y_py", by default "polar"
        indicator : string, optional
            dynamic indicator name, by default None
        *params : extra parameters for f

        Returns
        -------
        numpy array
            dynamic indicator values, flattened
        """        
        assert indicator is not None
        assert len(params) <= len(self.f.get_param_names())
        coords = self.compute_coords(extents, sampling, method)
        actual_params = list(params) + self.f.get_param_defaults()[len(params):len(self.f.get_param_defaults())]
        return self.f.compute_indicator(coords, n_turns, *actual_params, indicator_name=indicator, **kwargs)

    def compute_coords(self, extents, sampling, method="polar"):
        if method == "polar":
            x0 = np.linspace(extents[0], extents[1], sampling+2)[1:-1]
            y0 = np.linspace(extents[2], extents[3], sampling+2)[1:-1]
            xx, yy = np.meshgrid(x0, y0)
            xxf = xx.flatten() * np.cos(extents[4])
            pxf = xx.flatten() * np.sin(extents[4])
            yyf = yy.flatten() * np.cos(extents[5])
            pyf = yy.flatten() * np.sin(extents[5])
            return (xxf, pxf, yyf, pyf)
        elif method == "x_px":
            x0 = np.linspace(extents[0], extents[1], sampling+2)[1:-1]
            y0 = np.linspace(extents[2], extents[3], sampling+2)[1:-1]
            xx, yy = np.meshgrid(x0, y0)
            xxf = xx.flatten()
            pxf = yy.flatten()
            yyf = np.zeros_like(xxf)
            pyf = np.zeros_like(xxf)
            return (xxf, pxf, yyf, pyf)
        elif method == "y_py":
            x0 = np.linspace(extents[0], extents[1], sampling+2)[1:-1]
            y0 = np.linspace(extents[2], extents[3], sampling+2)[1:-1]
            xx, yy = np.meshgrid(x0, y0)
            yyf = xx.flatten()
            pyf = yy.flatten()
            xxf = np.zeros_like(yyf)
            pxf = np.zeros_like(yyf)
            return (xxf, pxf, yyf, pyf)

    def get_param_names(self):
        """Get the parameter names from the function wrapper

        Returns
        -------
        list
            list of parameter names
        """        
        return self.f.get_param_names()

    def get_param_defaults(self):
        """Get the parameter defaults from the function wrapper

        Returns
        -------
        list
            list of floats
        """        
        return self.f.get_param_defaults()

    def get_indicators_available(self):
        """Get the list of dynamic indicators available from the function wrapper

        Returns
        -------
        list
            list of strings
        """        
        return self.f.get_indicators_available()

    def get_indicator_param_names(self, name):
        return self.f.get_indicator_param_names(name)

    def get_indicator_param_defaults(self, name):
        return self.f.get_indicator_param_defaults(name)
