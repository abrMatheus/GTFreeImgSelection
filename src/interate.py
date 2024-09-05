import os
import numpy as np
import matplotlib.pyplot as plt
# from flim.experiments import utils
from skimage.color import gray2rgb
from skimage.filters import threshold_otsu
from sklearn.metrics.pairwise import euclidean_distances

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

from matplotlib.widgets import TextBox

import pandas as pd


import locale
import warnings
try:
    import pyift.pyift as ift
except ModuleNotFoundError:
    ift = None
    warnings.warn("pyift is not installed.", RuntimeWarning)



def load_mimage(path):
    assert ift is not None, "PyIFT is not available"

    locale.setlocale(locale.LC_NUMERIC, "en_US.UTF-8")

    mimage = ift.ReadMImage(path)

    mimage = mimage.AsNumPy().squeeze()

    return mimage.copy()



def load_image(path: str, lab: bool = True) -> np.ndarray:
    assert ift is not None, "PyIFT is not available"
    image = ift.ReadImageByExt(path)
    if ift.Is3DImage(image):
        if lab:
            mimage = ift.ImageToMImage(image, color_space=ift.LABNorm2_CSPACE)
            image = mimage.AsNumPy()
        else:
            image = image.AsNumPy()
        image = image.squeeze()
    else:  # 2D Image
        if lab:
            mimage = ift.ImageToMImage(image, \
                                             color_space=ift.LABNorm2_CSPACE)
            image = mimage.AsNumPy().transpose(0,1,2,3)
        else:
            image = image.AsNumPy()
        image = image.squeeze()
    return image




def get_descritor_act_all(svox_dir,im_list,act_dir,band, n0,nf, norm_type='pdf'):
    
    # act_list  = os.listdir(act_dir)
    # label_list = os.listdir(label_dir)
    
    des_array = np.zeros((nf, len(im_list)))
    print("len im list", len(im_list))
    
    for i, name in enumerate(im_list):
        iname = name.split(".mimg")[0]
        s_name = os.path.join(svox_dir,f"{iname}.png")
        a_name = os.path.join(act_dir,f"{iname}.mimg")
        
        if not os.path.exists(s_name):
            print(f"{s_name} does not exist!")
            continue
        if not os.path.exists(a_name):
            print(f"{a_name} does not exist!!")
            continue
        
        
        s_img = load_image(s_name, lab=False)
        a_img = load_mimage(a_name)
  
        
        # mudei aqui!!!!!
        #b_img = binary_by_otsu_kernel(a_img, s_img, band)
        
        
        b_img = a_img[:,:,band].copy()
    
    
        uniq, counts = np.unique(s_img, return_counts=True)
    
        for ni in range(nf):
            des_array[ni,i] = np.sum(b_img[s_img==ni])
            if norm_type=='norm_size' or norm_type=='norm_size_pdf':
                des_array[ni,i] = des_array[ni,i]/counts[ni]
                

            
    new_array = np.sort(des_array, axis=0)[::-1]
    
    if norm_type =="norm_size_pdf" or norm_type == "pdf":
        new_array = new_array/new_array.sum(0)
    
    return new_array.transpose()


def render_activation(act_img, orig_img, otsu=False, render_type='heatmap', islice=-1):
    #print(f"shapes {act_img.shape} {orig_img.shape}")
    #primeiro encontrar o slice do meio do bang

    # if islice==-1:
    
    #     limits=[300,-1]
        
    #     for i in range(act_img.shape[0]):
    #         if(act_img[i].sum()>2):
    #             #coordenada ativa
    #             if(limits[0]>i):
    #                 limits[0]=i
    #             if(limits[1]<i):
    #                 limits[1]=i
    #     thelimit = int(limits[0] + (limits[1]-limits[0])/2)
    # else:
    #     thelimit = islice
    
    # a_slice = act_img[thelimit]DONE 2D
    a_slice = act_img.copy()
    if otsu==True:
        thresh = threshold_otsu(a_slice)
        a_slice[a_slice<thresh]=0.0 #so zera valores abaixo
        
    a_slice = a_slice/(a_slice.max()+0.001) # de 0 a 1
    # o_slice = orig_img[thelimit] DONE 2D
    o_slice = orig_img.copy()
    o_slice = o_slice/(o_slice.max()+0.001) # de 0 a 1
    
    # o_slice = gray2rgb(o_slice) DONE 2D
    b_slice = 1-a_slice
    a_slice = gray2rgb(a_slice)
    b_slice = gray2rgb(b_slice)

    if render_type=='heatmap':
        o_slice = 0.5*o_slice + (1,0,0)*a_slice*0.5 + (0,0,1)*b_slice*0.5
        # o_slice = 0.45*o_slice + ((1,0,0)*a_slice*0.9 + (0,0,1)*b_slice*0.9)*0.55
    elif render_type == 'saliency':
        o_slice = a_slice


    #'heatmap','saliency', 'none'
    
    return o_slice




def mostra_acts(rec_min, rec_max, act_dir, im_dir, conf, otsu=False, nim=2, axs=None):
    band = conf[1]

    # agora irei mostrar essas ativações

    if axs is None:
        fig, axs = plt.subplots(2,nim, figsize=(15,7))
        fig.subplots_adjust(wspace=0.05, hspace=0.05)
    
    #fig, axs = plt.subplots(2,nim, figsize=(20,10))
    

    rec = rec_min
    for k in range(2): #2-> dist prox
        if k==1:
            rec = rec_max
        for i in range(nim):
            
            iname = rec[i].split(".")[0]
            
            a_path = os.path.join(act_dir,f"{iname}.mimg")
            o_path = os.path.join(im_dir,f"{iname}.png")
            
            if os.path.exists(a_path):
            
                print(f"a_path '{a_path}'")
                # print(f"o_path '{o_path}'")
                o_img = load_image(o_path,lab=False)
                a_img = load_mimage(a_path)
                # print(a_img.shape, o_img.shape, band)
                
                oslice = render_activation(a_img[:,:,band], o_img, otsu)
    
    
                axs[k][i].imshow(oslice)
                axs[k][i].set_xticks([])
                axs[k][i].set_yticks([])
            else:
                print("not exist", a_path)
            
            print(iname, end=",")
        print()
        #break
    axs[0][0].set_ylabel("Próximos")
    axs[1][0].set_ylabel("Distantes")
    #plt.show()

def obtem_imagens_proximas(desc_i, i_act_list, index_i, model='m1', nim=2):
    indexes_min = np.ones((2,nim))
    indexes_max = np.ones((2,nim))
    
    indexes_min[0] = np.ones((nim))*1000 # para salvar as distancias
    indexes_min[1] = np.ones((nim))*-1  # para salvar os indices
    
    indexes_max[0] = np.ones((nim))*0 # para salvar as distancias
    indexes_max[1] = np.ones((nim))*-1  # para salvar os indices
    
    desc_im = desc_i[index_i]
    
    for i in range(desc_i.shape[0]):
    
        if (i!=index_i):
            adist0 = euclidean_distances(desc_i[i].reshape(1, -1), desc_im.reshape(1, -1))
        
            if(adist0[0,0] < indexes_min[0,nim-1]):
                indexes_min[0,nim-1]=adist0[0,0]
                indexes_min[1,nim-1]=i
                
                
            if(adist0[0,0] > indexes_max[0,nim-1]):
                indexes_max[0,nim-1]=adist0[0,0]
                indexes_max[1,nim-1]=i
                
            sorti_max = np.flip(np.argsort(indexes_max[0], axis=0))
            sorti_min = np.argsort(indexes_min[0], axis=0)
                
            indexes_max[0] = np.take_along_axis(indexes_max[0], sorti_max, axis=0) 
            indexes_max[1] = np.take_along_axis(indexes_max[1], sorti_max, axis=0) 
        
            indexes_min[0] = np.take_along_axis(indexes_min[0], sorti_min, axis=0) 
            indexes_min[1] = np.take_along_axis(indexes_min[1], sorti_min, axis=0) 
        
        
    rec_min = []
    for i in range(nim):
        im_name = i_act_list[int(indexes_min[1][i])].split(".")[0]
        rec_min.append(f"{im_name}.mimg")

    rec_max = []
    for i in range(nim):
        im_name = i_act_list[int(indexes_max[1][i])].split(".")[0]
        # rec_ind = recImages['list'].index(im_name)
        # recImages[conf[0]][rec_ind] = [im_name, indexes_max[0][i], model, conf[1]]

        rec_max.append(f"{im_name}.mimg")



    print()      
    print('min',indexes_min)
    print(rec_min)
    print()
    print('max',indexes_max)
    print(rec_max)
    
    return indexes_min,indexes_max, rec_min, rec_max


def restart_RecImages():
    RecImages = {}

    for c in Classes:
        RecImages[c]=[]

    RecImages['list']=[]
    for i in origImList:
        for c in Classes:
            RecImages[c].append([i.split(".")[0], 0, 'na', -1])
        RecImages['list'].append(i.split(".")[0])
    return RecImages


def get_inner_conf(meta_model, inner_m):
    df = pd.read_csv(os.path.join(meta_model,inner_m,"k_description.csv"))
    conf = []
    for i,kc in enumerate(zip(df['kernel'], df['class'])):
        k=kc[0]
        c=kc[1]
        if c in Classes:
            conf.append((c, i))
    return conf


def add_new_model(inner_model, selected_image):
    # conf_inner_model = [("wt",3), ("et",2)] #configura os kernels do modelo [0,n]

    #conf---------
    conf_inner_model = get_inner_conf(meta_model,inner_model)
    global inner_act_dir
    inn_act_dir   = f"{meta_model}/{inner_model}/activation1"
    #conf---------
    
    ind_img = np.argwhere(act_list==selected_image)[0][0]
    index_sel.append(ind_img)


    # -------------------------------------------------------------- lendo e salvando confs
    for conf in conf_inner_model:
        pre_desc_file=f"{desc_folder}/desc_{inner_model}_{conf[0]}_{conf[1]}.npy"
        act_list_name=f"{desc_folder}/act_{inner_model}_{conf[0]}_{conf[1]}.npy"
    
        band = conf[1]
    
        if os.path.exists(pre_desc_file):
            desc_norm  = np.load(pre_desc_file)
            act_list_i = np.load(act_list_name)
    
        else:
            act_list_i  = np.array(act_list)
            desc_norm   = get_descritor_act_all(svox_dir,act_list_i,inn_act_dir,band, 100,20, norm_type='norm_size')
            with open(pre_desc_file, 'wb') as f:
                np.save(f, desc_norm)
            with open(act_list_name, 'wb') as f:
                np.save(f, np.array(act_list_i))  
    
        descs.append(desc_norm)
        #lists.append(act_list_i)
    
        if descs_class[conf[0]] is None:
            descs_class[conf[0]]=desc_norm
        else:
            descs_class[conf[0]] = np.concatenate((descs_class[conf[0]],desc_norm), axis=1)

    # -------------------------------------------------------------- lendo e salvando confs
    tmp_rec = {}
    tmp_emb = {}
    for i,c in enumerate(Classes):
        # print(i,c)
        
        tmp_rec[c] = obtem_imagens_proximas(descs_class[c], act_list, index_sel[-1])

        print(descs_class[c].shape)
        tmp_emb[c] = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=2).fit_transform(descs_class[c]) #perplexidade = n_amostras/n_classes que eu queria ver
        # tmp_emb[c] = umap.UMAP().fit_transform(descs_class[c])

        #tmp_rec[c] = obtem_imagens_proximas(tmp_emb[c], act_list, index_sel[-1])
    
    recs[inner_model] = tmp_rec
    x_embedded[inner_model] = tmp_emb
    
    conf_model_all[inner_model] = conf_inner_model
    
    # i_min1,i_max1, rec_min1, rec_max1


def get_image_by_index(name, act_dir, orig_dir, conf, render_type='heatmap', islice=-1, otsu=False):
    final_name = os.path.join(act_dir, name)
    iname = name.split(".")[0]
    o_path = os.path.join(orig_dir,f"{iname}.png")
    
    a_img = load_mimage(final_name)
    band = conf[1]
    o_img = load_image(o_path,lab=False)
                
    oslice = render_activation(a_img[:,:,band], o_img, otsu, render_type, islice) # DONE 2D
    print("render type", render_type)
    return oslice


def my_interact(nim=2):
    plt.close()

    def draw_axs(X_embedded, i_recs):
        axs[0].scatter(X_embedded[:,0],X_embedded[:,1], picker=True)
        axs[0].set_title("tSNE x1_1")
        
        
        markers = ['*', 'v', '>','1','2','3']
        colors  = ['black', 'red', 'aqua']
        
        for i, index in enumerate(index_sel):
            axs[0].scatter(X_embedded[index,0],X_embedded[index,1], marker=markers[i], color="black", label=act_list[index], s=200)
        
        
        axs[0].legend()
        
        
        count_aqua=0
        count_red=0
        for imin,imax in zip(i_recs[0][1], i_recs[1][1]):
            imin=int(imin)
            imax=int(imax)
            m = '2'
            for ii,c,m2,l2 in zip([imin,imax],['aqua', 'red'], ['*','v'], ['closest', 'distant']): 
                if count_aqua==0 and l2=='closest':
                    axs[0].scatter(X_embedded[ii,0],X_embedded[ii,1], marker=m2, color=c, label=l2)
                    count_aqua=1
                elif count_red ==0 and l2=='distant':
                    axs[0].scatter(X_embedded[ii,0],X_embedded[ii,1], marker=m2, color=c, label=l2)
                    count_red=1
                else:
                    axs[0].scatter(X_embedded[ii,0],X_embedded[ii,1], marker=m2, color=c)

        #TODO MOSTRAR MAIS DISTANTES

    
    global render_model
    render_model = list(recs.keys())[-1]
    global render_conf
    
    def render_f(c_render):
        global render_type
        render_type = c_render 
    
    def render_m(c_model):
        global render_model
        render_model = c_model 
    
    def render_c(c_conf):
        global render_conf
        render_conf = c_conf 

    def class_f(c_class):
        global new_class 
        new_class= c_class
        
    
    interact(render_f, c_render=['heatmap','saliency', 'none']);

    t_list = list(conf_model_all.keys())[::-1]
    interact(render_m, c_model=t_list);
    interact(render_c, c_conf=list(np.linspace(0,4,5)));


    #-----------------------------------------------------------_
    
    x = np.linspace(1,10,10)
    
    #%matplotlib widget
    
    # fig, axs = plt.subplots(2,1, figsize=(10,5))


    fig = plt.figure( figsize=(10,7))
    fig.subplots_adjust(wspace=0.05, hspace=0.1)

    gs = fig.add_gridspec(3,nim)
    ax_proj = fig.add_subplot(gs[0, 0:nim-1])
    axs_img = fig.add_subplot(gs[0,nim-1:])
    axs = [ax_proj, axs_img]

    
    axxs = []
    for i in range(2):
        axxs.append([])
        for j in range(nim):
            ax = fig.add_subplot(gs[i+1, j])
            axxs[i].append(ax)

    
    #script recomendação_tsne_iter-mult_run2_saving_displ.ipynb

    # iclass = 'wt'
    # conf=conf_model_all[inner_model][0]
    # mostra_acts(recs[inner_model][iclass][2], recs[inner_model][iclass][3], f"{meta_model}/{inner_model}/activation1", im_dirs[Classes.index(iclass)], conf, axs=axxs)

    iclass = 'obj'
    conf=conf_model_all[inner_model][0]
    mostra_acts(recs[inner_model][iclass][2], recs[inner_model][iclass][3], f"{meta_model}/{inner_model}/activation1", im_dirs[Classes.index(iclass)], conf, axs=axxs)

    print("here")

    ind =0
    def onpick3(event):
        # debug_msg = "click!!2"
        # os.system(f" echo {debug_msg} >> debug.txt")
        # im  = np.zeros((200,200))
        # im  = get_image_by_index(act_list[int(ind)], t_act_dir, modality_dir, conf, render_type)
        # get_image_by_index(act_list[0], inn_act_dir, im_dirs[1], conf_inner_model[0])
        # axs[1].imshow(im)
        if event.mouseevent.button == 1: #left
            ind = event.ind[0]
            # debug_msg = "click!!3"
            # os.system(f" echo {debug_msg} >> debug.txt")
            t_act_dir = f"{meta_model}/{render_model}/activation1"
            modality_dir = f"{imgs_dir}"

            if int(render_conf) < len(conf_model_all[render_model]):
                conf = conf_model_all[render_model][int(render_conf)]
                c = conf[0]
            else:
                conf = conf_model_all[render_model][0]
                c = conf[0]
            
            # im  = np.zeros((200,200))
            im  = get_image_by_index(act_list[int(ind)], t_act_dir, modality_dir, conf, render_type)
            # get_image_by_index(act_list[0], inn_act_dir, im_dirs[1], conf_inner_model[0])
            axs[1].imshow(im)
            axs[1].set_title(act_list[int(ind)])
            
            #TODO: clean no axs[0] e mostrar tudo de novo!!
    
            axs[0].clear()
            draw_axs(x_embedded[render_model][c], recs[render_model][c])
            axs[0].scatter(x_embedded[render_model][c][ind,0],x_embedded[render_model][c][ind,1], color="black", s=200)

            debug_msg = f"render {ind} {render_type} model {render_model} conf {render_conf} i {act_list[int(ind)]}"
            os.system(f" echo {debug_msg} >> debug.txt")
            
            # fig.canvas.draw()
    
    plt.ion()


    conf = conf_model_all[render_model][int(render_conf)-1]
    print(conf, render_conf, render_model)
    c = conf[0]
    draw_axs(x_embedded[render_model][c], recs[render_model][c])
    
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].set_xticks([])
    axs[1].set_yticks([])
        
    fig.canvas.mpl_connect('pick_event', onpick3)
    plt.show(block=True)
    
    
    #%matplotlib inline
    # plt.close()

#my_interact()




##############################################################################################################


label_dir = 'data/label'
data_dir  = 'data'
imgs_dir = 'data/orig' #usado para render
svox_dir  = 'data/supvox'

meta_model = 'data/flim_models'



Classes=['obj'] #,'bg'] #'nc', 'ed']
im_dirs = [imgs_dir] # o que fazer com o bg
IClasses = ['imgs']
origImList = os.listdir(label_dir)

if not os.path.exists(meta_model):
    os.system(f"mkdir {meta_model}")

desc_folder = f"{meta_model}/iter_selec_desc"
if not os.path.exists(desc_folder):
    os.system(f"mkdir {desc_folder}")

descs_class = {}
for c in Classes:
    descs_class[c]=None
recs = {}
x_embedded = {}
conf_model_all={}

descs = []
lists = []
index_sel = [] 





###################

inner_model = "m1"
selected_image = '000070.mimg'
act_list = np.array(os.listdir(f"{meta_model}/{inner_model}/activation1"))


add_new_model(inner_model, selected_image)




# #####################################

plt.close()

iclass = 'obj'
conf=conf_model_all[inner_model][0]
print("conf", conf)
# mostra_acts(recs[inner_model][iclass][2], recs[inner_model][iclass][3], f"{meta_model}/{inner_model}/activation1", im_dirs[Classes.index(iclass)], conf)



# ############


my_interact()
