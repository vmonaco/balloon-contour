'''
Created on Sep 18, 2012

@author: vinnie
'''
import sys
import pylab as plb
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import resample
from scipy.ndimage.filters import gaussian_filter
from math import floor, ceil, pi
from scipy.signal.signaltools import fftconvolve
from scipy.interpolate.fitpack2 import RectBivariateSpline

def ContourArea(p):
    "Compute the area of a contour. Negative for points going CW"
    o = np.concatenate([p, p[:,:2]], axis=1)
    area = 0.5*np.sum(o[0,1:p.shape[1]+1] * (o[1,2:p.shape[1]+2] - o[1,:p.shape[1]]))
    return area
    
def MakeClockwise(p):
    "Make a sequence of points clockwise if it isn't already"
    area = ContourArea(p)
    if area > 0:
        return p[:,::-1]
    else:
        return p
    
def ImageDerivatives2D(i, sigma, type):
    "Image derivatives with sigma"
    (x,y) = np.mgrid[floor(-3*sigma):ceil(3*sigma)+1,floor(-3*sigma):ceil(3*sigma)+1]
    
    if type is 'x':
        kernel = -(x/(2 * pi * sigma**4)) * np.exp(-(x**2 + y**2)/(2 * sigma**2))
    elif type is 'y':
        kernel = -(y/(2 * pi * sigma**4)) * np.exp(-(x**2 + y**2)/(2 * sigma**2))
    elif type is 'xx':
        kernel = 1/(2 * pi * sigma**4) * (x**2 / sigma**2 - 1) * np.exp(-(x**2 + y**2)/(2 * sigma**2))
    elif type is 'yy':
        kernel = 1/(2 * pi * sigma**4) * (y**2 / sigma**2 - 1) * np.exp(-(x**2 + y**2)/(2 * sigma**2))
    elif type in ['xy', 'yx']:
        kernel = 1/(2 * pi * sigma**6) * (x * y) * np.exp(-(x**2 + y**2)/(2 * sigma**2))
    
    return fftconvolve(i, kernel, 'same')

def ExternalForceImage2D(i, w_line, w_edge, w_term, sigma):
    "Creates the external forces (lines, edges) for the active contour"
    i_x = ImageDerivatives2D(i, sigma, 'x')
    i_y = ImageDerivatives2D(i, sigma, 'y')
    i_xx = ImageDerivatives2D(i, sigma, 'xx')
    i_yy = ImageDerivatives2D(i, sigma, 'yy')
    i_xy = ImageDerivatives2D(i, sigma, 'xy')
    e_line = gaussian_filter(i, sigma)
    
    e_term = (i_yy * i_x**2 - 2 * i_xy * i_x * i_y + i_xx * i_y**2)/((1 + i_x**2 + i_y**2)**(1.5))
    e_edge = np.sqrt(i_x**2 + i_y**2)
    e_extern = (w_line * e_line - w_edge * e_edge - w_term * e_term)
    
    return e_extern

def circshift(a, roll):
    "Roll a numpy array  more than once"
    assert len(roll) <= a.ndim
    for i in xrange(len(roll)):
        a = np.roll(a, roll[i], axis=i)
        
    return a
    
def SnakeInternalForceMatrix2D(n_points, alpha, beta, gamma):
    "Creates the internal forces of the active contour (balloon force)"
    b = np.empty(5)
    b[0] = beta
    b[1] = -(alpha  + 4*beta)
    b[2] = (2*alpha + 6*beta)
    b[3] = b[1]
    b[4] = b[0]
    
    A = b[0] * circshift(np.eye(n_points), (2,))
    A = A + b[1] * circshift(np.eye(n_points), (1,))
    A = A + b[2] * circshift(np.eye(n_points), (0,))
    A = A + b[3] * circshift(np.eye(n_points), (-1,))
    A = A + b[4] * circshift(np.eye(n_points), (-2,))
    
    return np.linalg.inv(A + gamma * np.eye(n_points))

def GVFOptimizeImageForces2D(f_ext, mu, iterations, sigma):
    "Gradient vector flow on a vector field"
    f_x = f_ext[0]
    f_y = f_ext[1]
    
    s_mag = f_x**2 + f_y**2
    
    u = f_x
    v = f_y
    
    for i in xrange(iterations):
        u_xx = ImageDerivatives2D(u, sigma, 'xx')
        u_yy = ImageDerivatives2D(u, sigma, 'yy')
        
        v_xx = ImageDerivatives2D(v, sigma, 'xx')
        v_yy = ImageDerivatives2D(v, sigma, 'yy')
        
        u = u + mu*(u_xx+u_yy) - s_mag * (u - f_x)
        v = v + mu*(v_xx+v_yy) - s_mag * (v - f_y)
    
    return np.array([u, v])

def GetContourNormals2D(p, a=4):
    "Get the normals of points on a contour"
    xt = p[0]
    yt = p[1]
    
    n = len(xt)
    f = np.arange(n) + a
    f[f>=n] = f[f>=n] - n
    b = np.arange(n) - a
    b[b<0] = b[b<0] + n
    
    dx = xt[f] - xt[b]
    dy = yt[f] - yt[b]
    
    l = np.sqrt(dx**2 + dy**2)
    nx = -dy/l
    ny = dx/l
    
    return np.array([nx,ny])

def SnakeMoveIteration2D(b, p, f_ext, gamma, kappa, delta):
    "Iterate the contour one step, using both internal and external forces."
    p[0] = np.minimum(np.maximum(p[0],1), f_ext[0].size)
    p[1] = np.minimum(np.maximum(p[1],1), f_ext[1].size)
    
    x_coords = np.arange(f_ext[0].shape[0])
    y_coords = np.arange(f_ext[0].shape[1])
    f_ext_x = RectBivariateSpline(x_coords, y_coords, f_ext[0])
    f_ext_y = RectBivariateSpline(x_coords, y_coords, f_ext[1])
    
    # get interpolated points evaluated at the contour points
    f_ext_1 = np.empty(shape=p.shape)
    f_ext_1[0] = kappa*f_ext_x.ev(p[0], p[1])
    f_ext_1[1] = kappa*f_ext_y.ev(p[0], p[1])
    
    n = GetContourNormals2D(p)
    f_ext_2 = delta * n
    
    ssx = gamma*p[0] + f_ext_1[0] + f_ext_2[0]
    ssy = gamma*p[1] + f_ext_1[1] + f_ext_2[1]
    
    p[0] = np.dot(b, ssx)
    p[1] = np.dot(b, ssy)
    
    p[0] = np.minimum(np.maximum(p[0],1), f_ext[0].size)
    p[1] = np.minimum(np.maximum(p[1],1), f_ext[1].size)
    
    return p

def InterpolateContourPoints2D(p, n_points):
    "Linear interpolation of a sequence of points. Smooths and interpolates"
    o = np.array([np.concatenate([p[0][-4:], p[0], p[0], p[0][:4]]),
                  np.concatenate([p[1][-4:], p[1], p[1], p[1][:4]])])
    
    n_samples = o.shape[1] * 10
    
    o = np.array([resample(o[0], n_samples),
                  resample(o[1], n_samples)])
    o = o[:,41:-38]
    dis = np.append(0,np.cumsum(np.sqrt(np.sum((o[:,1:]-o[:,:-1])**2, 0))))
    
    xi = np.linspace(0, dis[-1], n_points*2)
    y_1 = interp1d(dis,o[0], kind='linear')
    y_2 = interp1d(dis,o[1], kind='linear')
    k = np.array([y_1(xi), y_2(xi)])
    
    k_len = float(k.shape[1])
    k = k[:,round(k_len/4):round(k_len/4)+n_points]
    
    return k

def Snake2D(image, p, 
            n_points=100, 
            w_line=0.01, 
            w_edge=1, 
            w_term=0.01, 
            sigma1=40, 
            sigma2=20, 
            alpha=0.2, 
            beta=0.2, 
            delta=0.1,
            gamma=1,
            kappa=2,
            iterations=200,
            g_iterations=0,
            mu=0.2,
            sigma3=1):
    "Find the local optimum contour with given params"
    image = image.astype(np.float32)
    if image.ndim > 2:
        image = np.mean(image, axis=2)
    
    p = MakeClockwise(p)
    p = InterpolateContourPoints2D(p, n_points)
    
    e_ext = ExternalForceImage2D(image, w_line, w_edge, w_term, sigma1)
    f_x = ImageDerivatives2D(e_ext, sigma2, 'x')
    f_y = ImageDerivatives2D(e_ext, sigma2, 'y')
    
    f_ext = np.array([-f_x*2*sigma2**2, -f_y*2*sigma2**2])
    f_ext = GVFOptimizeImageForces2D(f_ext, mu, g_iterations, sigma3)
    
    s = SnakeInternalForceMatrix2D(n_points, alpha, beta, gamma)
    
    for i in xrange(iterations):
        print i
        p = SnakeMoveIteration2D(s, p, f_ext, gamma, kappa, delta)
        global _contour
        _contour.set_xdata(np.append(p[0,-1], p[0]))
        _contour.set_ydata(np.append(p[1,-1], p[1]))
        plb.draw()
    
    return p

def BalloonFromCircle(center, radius, image, n_points):
    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.array([radius*np.cos(theta), radius*np.sin(theta)])
    p = circle + np.array([[center[0]],[center[1]]])
    plb.ion()
    plb.figure(0)
    plb.gray()
    if image.ndim > 2:
        image = np.mean(image, axis=2)
    plb.imshow(image)
    global _contour
    _contour, = plb.plot(np.append(p[0,-1], p[0]),np.append(p[1,-1], p[1]))
    plb.draw()
    p = Snake2D(image, p)
    return InterpolateContourPoints2D(p, n_points)

def test_MakeClockwise(p):
    # make sure p is ccw
    area = ContourArea(p)
    if area < 0:
        p = p[:,::-1]
    p_r = MakeClockwise(p)
    assert (p == p_r[:,::-1]).all() # it actually got reversed
    return

def _test():
    # make a unit circle going counter-clockwise
    radius = 60
    theta = np.linspace(0, 2*np.pi, 10)
    circle_ccw = np.array([radius*np.cos(theta), radius*np.sin(theta)])
    area = ContourArea(circle_ccw)
    assert area > 0
    circle_cw = MakeClockwise(circle_ccw)
    area = ContourArea(circle_cw)
    assert area < 0
    assert (circle_cw == circle_ccw[:,::-1]).all() # it actually got reversed
    p = circle_cw + np.array([[280],[430]])

    plb.ion()
    plb.figure(0)
    plb.gray()
    i = plb.imread('mri2.png')
    i = np.mean(i, axis=2)
    plb.imshow(i)
    global _contour
    _contour, = plb.plot(np.append(p[0,-1], p[0]),np.append(p[1,-1], p[1]))
    plb.draw()
    Snake2D(i, p, iterations=500)
    print 'done'
    plb.ioff()
    plb.savefig('mri-result.png')
    plb.show()
    
def _run(filename):
    
    return
    
if __name__ == "__main__":
    if (len(sys.argv) > 1):
        _run(sys.argv[1])
    else:
        _test();