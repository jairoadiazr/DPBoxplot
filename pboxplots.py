import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#Required functions for different quantile estimation methods
import diffprivlib #PrivateQuantile
from private_quantiles.approximate_quantiles_algo import approximate_quantiles_algo #ApproxQuantile
from private_quantiles.joint_exp import joint_exp #JointExp
from private_quantiles.unboundedQuantile import unboundedQuantile #unbounded


#MAIN FUNCTIONS

#sns wrapper for private boxplot
def pboxplot(data=None, x=None, y=None, eps = 5, w=7/8, bounds = (-50,50), seed=-1, method = 'DPBoxplot', hue=None, order=None, hue_order=None, orient=None, color=None, palette=None, saturation=0.75, fill=True, dodge='auto', width=0.8, gap=0, whis=1.5, linecolor='auto', linewidth=None, fliersize=None, hue_norm=None, native_scale=False, log_scale=None, formatter=None, legend='auto', ax=None, **kwargs):
    
    add_hue = False
    
    if seed>=0:
        np.random.seed(seed)
        
    df_pboxplots = []
    df_pboxplots_info = []
        
    if (x is None) and (y is None):
    
        if not(hue is None):
            raise Exception("Cannot use `hue` without `x` and `y`")
            
        df = data.select_dtypes(include=['number'])
                
        for column in df.columns:
            cur_x = list(df[column])
            keps = 1
            box_whiskers, outliers, whiskers_size, n = get_box_plot(cur_x, dp = True, eps = eps/keps, w = w, bounds = bounds, method = method)
            df_pboxplots.append(box_whiskers)
            df_pboxplots_info.append(outliers + [whiskers_size, n])
        
        df_pboxplots = pd.DataFrame(df_pboxplots).T
        df_pboxplots.columns = df.columns
        df_pboxplots_info = pd.DataFrame(df_pboxplots_info).T
        df_pboxplots_info.columns = df.columns
        
        df_pboxplots = df_pboxplots[list(np.sort(df_pboxplots.columns))]
        df_pboxplots_info = df_pboxplots_info[list(np.sort(df_pboxplots_info.columns))]
        sns.boxplot(data=df_pboxplots, ax = ax)
    
    else:
        if x is None:
            cur_x = list(data[y])
            box_whiskers, outliers, whiskers_size, n = get_box_plot(cur_x, dp = True, eps = eps, w = w, bounds = bounds, method = method)
            df_pboxplots.append(box_whiskers)
            df_pboxplots_info.append(outliers + [whiskers_size, n])
            colnames = [y]
            
            df_pboxplots = pd.DataFrame(df_pboxplots).T 
            df_pboxplots.columns = colnames
            df_pboxplots_info = pd.DataFrame(df_pboxplots_info).T
            df_pboxplots_info.columns = colnames
            
            df_pboxplots = df_pboxplots[list(np.sort(df_pboxplots.columns))]
            df_pboxplots_info = df_pboxplots_info[list(np.sort(df_pboxplots_info.columns))]
            
            sns.boxplot(data=df_pboxplots, ax = ax)
            
        else:
        
            datacopy = data.copy()

            
            if hue is None:
                hue = 'hue'
                datacopy[hue] = 1
            else:
                add_hue = True
                
            try:
                cat_hue = list(datacopy[hue].cat.categories)
            except:
                try:
                    cat_hue = datacopy[hue].unique()
                except:
                    cat_hue = np.unique(datacopy[hue])
            
            try:
                cat_x = list(datacopy.drop(columns = hue)[x].cat.categories)
            except:
                try:
                    cat_x = datacopy.drop(columns = hue)[x].unique()
                except:
                    cat_x = np.unique(datacopy.drop(columns = hue)[x])
            
            cat_x = np.sort(cat_x)
            cat_hue = np.sort(cat_hue)
            colnames = cat_x
            
            numboxplots = 0
            sumlog = 0
            for cur_cat_x in colnames:
                for cur_cat_hue in cat_hue:
                    cur_x = list(datacopy[(datacopy[x]==cur_cat_x)&(datacopy[hue]==cur_cat_hue)][y])
                    numboxplots = numboxplots + 1
                    sumlog = sumlog + np.log(len(cur_x))
                              
            for cur_cat_x in colnames:
                
                df_box_whiskers = []
                df_info_boxplot = []
                
                hue_column = []
                hue_column_info = []
                cc_info = []
                
                for cur_cat_hue in cat_hue:
                    cur_x = list(datacopy[(datacopy[x]==cur_cat_x)&(datacopy[hue]==cur_cat_hue)][y])
                    

                    #keps = sumlog/np.log(len(cur_x))
                    keps = 1
                    box_whiskers, outliers, whiskers_size, n = get_box_plot(cur_x, dp = True, eps = eps/keps, w = w, bounds = bounds, method = method)
                    
                    df_box_whiskers = df_box_whiskers + box_whiskers
                    hue_column = hue_column + [cur_cat_hue for i in box_whiskers]
                    
                    info_boxplot = outliers + [whiskers_size, n]
                    df_info_boxplot = df_info_boxplot + info_boxplot
                    hue_column_info = hue_column_info + [cur_cat_hue for i in info_boxplot]
                    cc_info = cc_info + ['l_out','u_out','whiskers_size','n']
                    
                    #print([cur_cat_x, cur_cat_hue, outliers[0]*n, outliers[1]*n])
                    
                df_box_whiskers = pd.DataFrame([df_box_whiskers]).T 
                df_box_whiskers.columns = [y]
                df_box_whiskers[x] = cur_cat_x
                df_box_whiskers[hue] = hue_column
                
                df_info_boxplot = pd.DataFrame([df_info_boxplot]).T 
                df_info_boxplot.columns = [y]
                df_info_boxplot[x] = cur_cat_x
                df_info_boxplot[hue] = hue_column_info
                df_info_boxplot['cc'] = cc_info
                
                df_pboxplots.append(df_box_whiskers)
                df_pboxplots_info.append(df_info_boxplot)
                
            df_pboxplots = pd.concat(df_pboxplots).reset_index(drop = True)
            df_pboxplots_info = pd.concat(df_pboxplots_info).reset_index(drop = True)
            
            if len(cat_hue) > 1:
                if ax is None:
                    ax = sns.boxplot(data=df_pboxplots, x=x, y=y, hue=hue)
                else:
                    sns.boxplot(data=df_pboxplots, x=x, y=y, hue=hue, ax = ax)
                
                n_out_low = np.array(df_pboxplots_info[df_pboxplots_info['cc']=='l_out'][y])*np.array(df_pboxplots_info[df_pboxplots_info['cc']=='n'][y])
                n_out_low = [int(np.max([0,np.ceil(i)])) for i in n_out_low]
                
                n_out_up = np.array(df_pboxplots_info[df_pboxplots_info['cc']=='u_out'][y])*np.array(df_pboxplots_info[df_pboxplots_info['cc']=='n'][y])
                n_out_up = [int(np.max([0,np.ceil(i)])) for i in n_out_up]
                
                whisker_max = np.max(df_pboxplots_info[df_pboxplots_info['cc']=='whiskers_size'][y])
                
                add_boxplot_labels(ax, n_out_low, n_out_up, whisker_max)
                
                return None
                
            else:
                df_pboxplots_info_temp = pd.DataFrame()
                df_pboxplots_temp = pd.DataFrame()
                for cur_cat_x in np.unique(df_pboxplots_info[x]):

                    df_pboxplots_info_temp[cur_cat_x] = list(df_pboxplots_info[df_pboxplots_info[x]==cur_cat_x][y])
                    df_pboxplots_temp[cur_cat_x] = list(df_pboxplots[df_pboxplots[x]==cur_cat_x][y])
                    
                df_pboxplots_info = df_pboxplots_info_temp
                df_pboxplots = df_pboxplots_temp
                
                sns.boxplot(data=df_pboxplots, ax = ax)

    coord = 0

    cur_min= np.Inf
    cur_max = -np.Inf
    whisker_max = -np.Inf

    for column in df_pboxplots_info.columns:
        p_bp_info = list(df_pboxplots_info[column])

        if p_bp_info[2] > whisker_max:
            whisker_max = p_bp_info[2]

    for column in df_pboxplots_info.columns:
        p_bp = list(df_pboxplots[column])
        p_bp_info = list(df_pboxplots_info[column])

        n = p_bp_info[3]

        if (p_bp[0] - whisker_max*0.15) < cur_min:
            cur_min = p_bp[0] - whisker_max*0.15

        if (p_bp[4] + whisker_max*0.15) > cur_max:
            cur_max = p_bp[4] + whisker_max*0.15

        n_out_low = int(np.max([0, np.ceil(n*p_bp_info[0])]))
        n_out_up = int(np.max([0, np.ceil(n*p_bp_info[1])]))
        
        if ax is None:
            plt.annotate('\n' + str(n_out_low), (coord, p_bp[0]), ha='center', va='top', color='black', size=10)
            plt.annotate(str(n_out_up) + '\n', (coord, p_bp[4]), ha='center', va='bottom', color='black', size=10)
        else:
            ax.annotate('\n' + str(n_out_low), (coord, p_bp[0]), ha='center', va='top', color='black', size=10)
            ax.annotate(str(n_out_up) + '\n', (coord, p_bp[4]), ha='center', va='bottom', color='black', size=10)
        coord = coord+ 1

    if ax is None:
        plt.ylim(cur_min, cur_max)
        
        if not(x is None):
            plt.xlabel(x)
        if not(y is None):
            plt.ylabel(y)
    else:
        ax.set_ylim(cur_min, cur_max)
        
        if not(x is None):
            ax.set_xlabel(x)
        if not(y is None):
            ax.set_ylabel(y)

#get differentially private boxplot
def get_box_plot(x, dp = True, eps = 10, w=7/8, bounds = (-50,50), method = 'DPBoxplot', box_method = 'JointExp', b=1.001, swap = False, c = 20, delta = 0.25):

    # INPUT
    # x --> data (vector)
    # dp --> True if calculating private boxplot
    
    # input only if dp==True:
        # eps --> epsilon for dp
        # w --> weights to distribute privacy budget: 
                #w[0]: quantile caculations (box and whiskers)
                #w[1]: outliers
        # bounds --> data bounds
        # bounds --> data bounds
    
    #OUTPUT
    # box_whiskers --> five points for drawing boxplot (lower whisker, lower box bound, median, upper box bound, upper whisker
    # outliers --> proportion of lower outliers and upper outliers
    # whiskers size -- difference between upper and lower whisker
    # n -- length of data
    
    eps_bp = eps*w
    eps_outliers = eps*(1-w)

    n = len(x)
    
    extreme_quantile = 1/np.sqrt(n)/c
    #extreme_quantile = 1/n*c
    
    if dp==True:
        #calculate points for private boxplot
        
        if method == 'PrivateQuantile':
            
            #calculate dp minimum
            try:
                qmin = diffprivlib.tools.quantile(x, extreme_quantile, epsilon=eps_bp*1/5, bounds = bounds)
            except:
                qmin = -np.Inf
                
            try:
                qmax = diffprivlib.tools.quantile(x, 1-extreme_quantile, epsilon=eps_bp*1/5, bounds = bounds)
            except:
                qmax = np.Inf
                
            lb = diffprivlib.tools.quantile(x, 0.25, epsilon=eps_bp*1/5, bounds = bounds)
            med = diffprivlib.tools.quantile(x, 0.5, epsilon=eps_bp*1/5, bounds = bounds)
            ub = diffprivlib.tools.quantile(x, 0.75, epsilon=eps_bp*1/5, bounds = bounds)
        
        elif method == 'unbounded':
            
            #calculate dp minimum
            try:
                try:
                    qmin = -unboundedQuantile(-x, -bounds[1], b=b, q = 1-extreme_quantile, eps = eps_bp*1/5)
                except:
                    qmin = unboundedQuantile(x, bounds[0], b=b, q = extreme_quantile, eps = eps_bp*1/5)
            except:
                qmin = -np.Inf
                
            try:
                qmax = unboundedQuantile(x, bounds[0], b=b, q = 1-extreme_quantile, eps = eps_bp*1/5)
            except:
                qmax = np.Inf
            
            try:
                lb = -unboundedQuantile(-x, -bounds[1], b=b, q = 0.75, eps = eps_bp*1/5)
            except:
                lb = unboundedQuantile(x, bounds[0], b=b, q = 0.25, eps = eps_bp*1/5)
            
            med = unboundedQuantile(x, bounds[0], b=b, q = 0.5, eps = eps_bp*1/5)
            ub = unboundedQuantile(x, bounds[0], b=b, q = 0.75, eps = eps_bp*1/5)
            
        elif method == 'JointExp':
            qmin,lb,med,ub,qmax = joint_exp(np.sort(x),bounds[0],bounds[1], qs = np.array([extreme_quantile,0.25,0.5,0.75,1-extreme_quantile]), eps = eps_bp, swap=swap)
            
        elif method == 'ApproxQuantile':
            qmin,lb,med,ub,qmax = approximate_quantiles_algo(x, np.array([extreme_quantile,0.25,0.5,0.75,1-extreme_quantile]), bounds = [bounds[0],bounds[1]], epsilon = eps_bp, swap=swap)
            
        if method == 'DPBoxplot':
            #calculate dp minimum
            try:
                if len(x)<1000:
                    print('warning')
                    raise Exception("Data size not appropriate for unbound estimation")
                try:
                    qmin = -unboundedQuantile(-x, -bounds[1], b=b, q = 1-extreme_quantile, eps = eps_bp*1/5)
                except:
                    qmin = unboundedQuantile(x, bounds[0], b=b, q = extreme_quantile, eps = eps_bp*1/5)
                qmax = unboundedQuantile(x, bounds[0], b=b, q = 1-extreme_quantile, eps = eps_bp*1/5)
                
                if qmax>qmin:
                    if box_method == 'JointExp':
                        lb,med,ub = joint_exp(np.sort(x), bounds[0], bounds[1], qs = np.array([0.25,0.5,0.75]), eps = eps_bp*3/5, swap=swap)
                    elif box_method == 'ApproxQuantile':
                        lb,med,ub = approximate_quantiles_algo(x, np.array([0.25,0.5,0.75]), bounds = [bounds[0],bounds[1]], epsilon = eps_bp*3/5, swap=swap)
                else:
                    raise Exception("Private upper bound is lower than private lower bound")
            except:
                #print("Warning: Error calculating private lower and/or upper bounds")
                if box_method == 'JointExp':
                    qmin,lb,med,ub,qmax = joint_exp(np.sort(x),bounds[0],bounds[1], qs = np.array([extreme_quantile,0.25,0.5,0.75,1-extreme_quantile]), eps = eps_bp, swap=swap)
                elif box_method == 'ApproxQuantile':
                    qmin,lb,med,ub,qmax = approximate_quantiles_algo(x, np.array([extreme_quantile,0.25,0.5,0.75,1-extreme_quantile]), bounds = [bounds[0],bounds[1]], epsilon = eps_bp, swap=swap)
        
      
        qmax = np.min([qmax, bounds[1]])
        qmin = np.max([qmin, bounds[0]])
        
        ub = np.min([ub, qmax])
        lb = np.max([lb, qmin])
            
        lw, uw, low_out, up_out = get_whiskers_outliers_points(x, qmin, lb, med, ub, qmax, n, eps_outliers, w, bounds = bounds, delta = delta)
    
    else:
        #calculate points for NON-private boxplot
        lb=np.percentile(x, 25)
        med=np.percentile(x, 50)
        ub=np.percentile(x, 75)

        lw = lb-1.5*(ub-lb)
        uw = ub+1.5*(ub-lb)
        
        if lw < np.min(x):
            lw = np.min(x)
        
        if uw > np.max(x):
            uw = np.max(x)
        
        low_out=np.mean(x<lw)
        up_out=np.mean(x>uw)
        
    whiskers_size = uw - lw
    
    box_whiskers = [lw,lb,med,ub,uw]
    box_whiskers=[float(x) for x in box_whiskers]
    
    outliers = [low_out, up_out]
    outliers=[float(x) for x in outliers]
    
    return box_whiskers, outliers, whiskers_size, n
 
#get whiskers and number of outliers
def get_whiskers_outliers_points(x, qmin, lb, med, ub, qmax, n, eps = 1, w = None, bounds = (-50,50), delta = 0):
    
    #Calculate whiskers and outlier points based on outlier proportion for better visualization of outliers and skewness
    n = len(x)
    
    lw = lb-1.5*(ub-lb)-0
    uw = ub+1.5*(ub-lb)+0
    
    
    if (qmin > lw) and (abs((qmin-lw)/lw) > (n**(-1/2+delta))):
        lw = np.min([qmin,lb])
        low_out = 0 #no low outliers when estimated minimum is higher than estimated lower whisker
    else:
        lw = np.max([lw, bounds[0]])
        #estimate dp outliers if the estimated minimum is lower than lower whisker
        low_out=np.mean(x<lw)+np.random.laplace(loc=0, scale=1, size=1)/(n*eps/2)

        
    #check if maximum differs 10% from upper whisker, if so, then modify whisker accordingly
    if (qmax < uw) and (abs((qmax-uw)/uw) > (n**(-1/2+delta))):
        uw = np.max([qmax,ub])
        up_out = 0 #no upper outliers when estimated maximum is lower than estimated upper whisker
    else:
        uw = np.min([uw, bounds[1]])
        #estimate dp outliers if the estimated maximum is higher than upper whisker
        up_out=np.mean(x>uw)+np.random.laplace(loc=0, scale=1, size=1)/(n*eps/2)
    

    
    return lw, uw, low_out, up_out


#AUXILIARY FUNCTIONS

#add outliers to private boxplot
def add_boxplot_labels(ax, n_out_low, n_out_up, whisker_max, fmt='.0f'):
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    
    cur_min = np.Inf
    cur_max = -np.Inf
    
    ii=0
    # Adjusted loop to focus on the upper whisker
    for i in range(2, len(lines), lines_per_box):
  
        upper_whisker = lines[i+1]
        lower_whisker = lines[i]
        
        #UPPER
        x, y = (data.mean() for data in upper_whisker.get_data())
        value = n_out_up[ii]
        
        y = y + whisker_max*0.05
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='bottom', 
                       color='black', size=10)
        if y + whisker_max*0.10 > cur_max:
            cur_max = y + whisker_max*0.10
            
        #LOWER
        x, y = (data.mean() for data in lower_whisker.get_data())
        value = n_out_low[ii]
        
        y = y - whisker_max*0.05
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='top',
                       color='black', size=10)
        
        if y - whisker_max*0.05 < cur_min:
            cur_min = y - whisker_max*0.05
            
        ii = ii + 1
    
    ax.set_ylim([cur_min,cur_max])

# Takes two boxplots x and y and computes the distance between them. 
def box_plot_distance(X, Y):
    
    def norm(x,y):
        z = np.array(x)-np.array(y)
        result = np.sum(np.abs(z))
        return result
    
    x = np.array(X[0])
    y = np.array(Y[0])
    outliers_x = np.array(X[1])
    outliers_y = np.array(Y[1])
    
    d_distance = norm(x,y)
    d2_distance = norm(x-x[2],y-y[2])
    
    iqr_distance = norm(x[1]-x[3], y[1]-y[3])
    m_distance = norm(x[2],y[2])
    w_distance = norm([x[0]-x[1], x[3]-x[4]], [y[0]-y[1], y[3]-y[4]])
    o_distance = norm(outliers_x, outliers_y)
        
    order_distance = float(np.max((y[1]<x[0],x[1]<y[0])) + np.max((y[2]<x[1],x[2]<y[1])) + np.max((y[2]>x[3],x[2]>y[3])) + np.max((y[3]>x[4],x[3]>y[4])))/4
        
    return [d_distance, d2_distance, iqr_distance, m_distance, w_distance, o_distance, order_distance]

