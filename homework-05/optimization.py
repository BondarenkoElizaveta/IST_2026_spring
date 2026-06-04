import numpy as np
from numpy.linalg import LinAlgError
import scipy
from datetime import datetime
from collections import defaultdict
try:
    from scipy.optimize._linesearch import scalar_search_wolfe2
except ImportError:  # pragma: no cover
    from scipy.optimize.linesearch import scalar_search_wolfe2


class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """
        if self._method == 'Constant':
            return self.c

        phi = lambda alpha: oracle.func_directional(x_k, d_k, alpha)
        derphi = lambda alpha: oracle.grad_directional(x_k, d_k, alpha)

        def armijo_search(start_alpha):
            alpha = start_alpha
            phi_0 = phi(0.0)
            derphi_0 = derphi(0.0)
            while phi(alpha) > phi_0 + self.c1 * alpha * derphi_0:
                alpha *= 0.5
                if alpha <= 0 or not np.isfinite(alpha):
                    return None
            return alpha

        if self._method == 'Armijo':
            alpha_0 = previous_alpha if previous_alpha is not None else self.alpha_0
            return armijo_search(alpha_0)

        if self._method == 'Wolfe':
            try:
                alpha = scalar_search_wolfe2(phi, derphi, c1=self.c1, c2=self.c2)[0]
            except Exception:
                alpha = None
            if alpha is not None and np.isfinite(alpha):
                return alpha
            return armijo_search(self.alpha_0)

        return None


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradien descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options={'method': 'Armijo', 'c1': 1e-4})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0).astype(float)
    start_time = datetime.now()
    previous_alpha = None

    def update_history(x, f_value, grad_norm):
        if trace:
            history['time'].append((datetime.now() - start_time).total_seconds())
            history['func'].append(f_value)
            history['grad_norm'].append(grad_norm)
            if x.size <= 2:
                history['x'].append(np.copy(x))

    try:
        f_k = oracle.func(x_k)
        g_k = oracle.grad(x_k)
        grad_norm_k = np.linalg.norm(g_k)
    except Exception:
        return x_k, 'computational_error', history

    if not np.all(np.isfinite(x_k)) or not np.all(np.isfinite(f_k)) or not np.all(np.isfinite(g_k)):
        return x_k, 'computational_error', history

    grad_norm_0_sq = grad_norm_k ** 2
    update_history(x_k, f_k, grad_norm_k)

    if display:
        print('Gradient descent: iter = 0, f = {}, grad_norm = {}'.format(f_k, grad_norm_k))

    if grad_norm_0_sq == 0 or grad_norm_k ** 2 <= tolerance * grad_norm_0_sq:
        return x_k, 'success', history

    for k in range(max_iter):
        d_k = -g_k
        alpha_start = None
        if previous_alpha is not None and line_search_tool._method == 'Armijo':
            alpha_start = 2.0 * previous_alpha
        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=alpha_start)
        if alpha is None or not np.isfinite(alpha):
            return x_k, 'computational_error', history

        x_k = x_k + alpha * d_k
        previous_alpha = alpha

        try:
            f_k = oracle.func(x_k)
            g_k = oracle.grad(x_k)
            grad_norm_k = np.linalg.norm(g_k)
        except Exception:
            return x_k, 'computational_error', history

        if (not np.all(np.isfinite(x_k)) or not np.all(np.isfinite(f_k))
                or not np.all(np.isfinite(g_k))):
            return x_k, 'computational_error', history

        update_history(x_k, f_k, grad_norm_k)
        if display:
            print('Gradient descent: iter = {}, f = {}, grad_norm = {}'.format(k + 1, f_k, grad_norm_k))

        if grad_norm_k ** 2 <= tolerance * grad_norm_0_sq:
            return x_k, 'success', history

    return x_k, 'iterations_exceeded', history


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    """
    Newton's optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively. If the Hessian
        returned by the oracle is not positive-definite method stops with message="newton_direction_error"
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'newton_direction_error': in case of failure of solving linear system with Hessian matrix (e.g. non-invertible matrix).
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = newton(oracle, np.zeros(5), line_search_options={'method': 'Constant', 'c': 1.0})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0).astype(float)
    start_time = datetime.now()

    def update_history(x, f_value, grad_norm):
        if trace:
            history['time'].append((datetime.now() - start_time).total_seconds())
            history['func'].append(f_value)
            history['grad_norm'].append(grad_norm)
            if x.size <= 2:
                history['x'].append(np.copy(x))

    try:
        f_k = oracle.func(x_k)
        g_k = oracle.grad(x_k)
        grad_norm_k = np.linalg.norm(g_k)
    except Exception:
        return x_k, 'computational_error', history

    if not np.all(np.isfinite(x_k)) or not np.all(np.isfinite(f_k)) or not np.all(np.isfinite(g_k)):
        return x_k, 'computational_error', history

    grad_norm_0_sq = grad_norm_k ** 2
    update_history(x_k, f_k, grad_norm_k)

    if display:
        print('Newton: iter = 0, f = {}, grad_norm = {}'.format(f_k, grad_norm_k))

    if grad_norm_0_sq == 0 or grad_norm_k ** 2 <= tolerance * grad_norm_0_sq:
        return x_k, 'success', history

    for k in range(max_iter):
        try:
            hess = oracle.hess(x_k)
            if scipy.sparse.issparse(hess):
                hess = hess.toarray()
            hess = np.asarray(hess, dtype=float)
            if hess.ndim == 1:
                hess = np.diag(hess)
            cho = scipy.linalg.cho_factor(hess)
            d_k = scipy.linalg.cho_solve(cho, -g_k)
        except (LinAlgError, scipy.linalg.LinAlgError, ValueError):
            return x_k, 'newton_direction_error', history
        except Exception:
            return x_k, 'newton_direction_error', history

        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=None)
        if alpha is None or not np.isfinite(alpha):
            return x_k, 'computational_error', history

        x_k = x_k + alpha * d_k

        try:
            f_k = oracle.func(x_k)
            g_k = oracle.grad(x_k)
            grad_norm_k = np.linalg.norm(g_k)
        except Exception:
            return x_k, 'computational_error', history

        if (not np.all(np.isfinite(x_k)) or not np.all(np.isfinite(f_k))
                or not np.all(np.isfinite(g_k))):
            return x_k, 'computational_error', history

        update_history(x_k, f_k, grad_norm_k)
        if display:
            print('Newton: iter = {}, f = {}, grad_norm = {}'.format(k + 1, f_k, grad_norm_k))

        if grad_norm_k ** 2 <= tolerance * grad_norm_0_sq:
            return x_k, 'success', history

    return x_k, 'iterations_exceeded', history
