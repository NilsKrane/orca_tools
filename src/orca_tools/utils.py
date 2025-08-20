'''Some util functions for jupyter notebooks'''

# ------------------------------------------------------------------------------------------

def molplot(molecule: str, vibs: bool=False):
    '''Substitute this cell with template to display molecule, using the py3Dmol package.

    :param molecule: String containing name of molecule instance
    :type molecule: str
    :param vibs: Animate a vibrational mode, defaults to False
    :type vibs: bool, optional
    '''
    content = ["#import py3Dmol\n"]
    content += ["view = py3Dmol.view(width=400, height=400)"]
    if vibs:
        content += [f"view.addModel({molecule}"+".xyz_vibration_string(0,amplitude=1) , 'xyz', {'vibrate': {'frames':20,'amplitude':1.0}})"]
        content += ["view.animate({'loop': 'backandforth', 'interval':20})"]
        content += ["view.rotate(-45,'x')"]
    else:
        content += [f"view.addModel({molecule}"+".xyz_string() , 'xyz')"]

    content += ["view.setStyle({'stick': {'radius': 0.15}, 'sphere': {'scale': 0.3}})"]
    content += ["view.zoomTo()"]
    content += ["view.show()"]

    __create_new_cell('\n'.join(content),replace=False)    

# ------------------------------------------------------------------------------------------

def cubeplot(cube: str):
    '''Substitute this cell with template to display cube as iso-surface, using the py3Dmol package.

    :param cube: String containing name of cube instance
    :type cube: str
    '''
    content = ["#import py3Dmol\n"]
    content += ["view = py3Dmol.view(width=400, height=400)"]
    content += [f"view.addModel({cube}"+".xyz_string() , 'xyz')"]
    content += [f"view.addVolumetricData({cube}"+".content, 'cube', {'isoval': -0.01, 'color': 'lightblue', 'opacity': 0.95})"]
    content += [f"view.addVolumetricData({cube}"+".content, 'cube', {'isoval': 0.01, 'color': 'pink', 'opacity': 0.95})"]
    content += ["view.rotate(-30,'x')"]


    content += ["view.setStyle({'stick': {'radius': 0.15}, 'sphere': {'scale': 0.3}})"]
    content += ["view.zoomTo()"]
    content += ["view.show()"]

    __create_new_cell('\n'.join(content),replace=False)    

# ------------------------------------------------------------------------------------------

def __create_new_cell(contents: str, replace: bool):
    '''Create new cell in jupyter notebook.

    :param contents: Fill cell with string content
    :type contents: str
    :param replace: If True replace selected cell with content. Otherwise create new cell below
    :type replace: bool
    '''    
    try:
        from IPython.core.getipython import get_ipython
    except ImportError:
        return None

    shell = get_ipython()
    payload = dict(
        source='set_next_input',
        text=contents,
        replace=replace,
    )
    shell.payload_manager.write_payload(payload, single=False)

