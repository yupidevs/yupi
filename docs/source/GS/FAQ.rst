Frequently Asked Questions
==========================

1. Why are plots not showing up? (Qt core dumped error)
-------------------------------------------------------

If you get an error similar to this::

   QObject::moveToThread: Current thread (0x5632706342d0) is not the object's thread (0x5632
   706194c0).
   Cannot move to target thread (0x5632706342d0)
   
   qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "/home/USERNAME/.local/lib/p
   ython/site-packages/cv2/qt/plugins" even though it was found.
   This application failed to start because no Qt platform plugin could be initialized. Rein
   stalling the application may fix this problem.
   
   Available platform plugins are: xcb, eglfs, linuxfb, minimal, minimalegl, offscreen, vnc,
    wayland-egl, wayland, wayland-xcomposite-egl, wayland-xcomposite-glx, webgl.
   
   [1]    1498499 IOT instruction (core dumped)  python example.py

it's proably because you have **PyQt5** and **opencv-python** packages
installed in the same python enviroment. This is due to a known incompatibility
between **PyQt5** and **opencv-python** in Linux systems. 

Solutions
+++++++++

Removing **PyQt5**
******************

If you don't need **PyQt5** in the current enviroment (globally by default) you
can remove it by simply running:

.. code-block:: bash
   
    $ pip uninstall PyQt5

Runing **yupi** in a separated virtual enviroment
*************************************************

If you don't want to remove **PyQt5** globally, you can create a separated
enviroment with **yupi**. This can be achieved in an easy way by using the
`venv` tool provided by python. You can visit `here
<https://docs.python.org/3/library/venv.html#module-venv>`_ the section
dedicated to this tool in the Python Docs which contains a guide for the
enviroment creation and activation.

Once the enviroment is created and activated you can install **yupi** (and all
the dependencies you might need) using the `pip` command.

.. code-block:: bash

    $ pip install yupi

.. note::

    The last solutions assume that **PyQt5** is not required for your project.
    If **PyQt5** is one of your project dependencies the next solution shows
    a way to use it along with **yupi**.

Install opencv from a linux repository
**************************************

If **PyQt5** is also needed along with **yupi** a known solution is to install
**opencv** from a linux repository. First you need to remove the current
**opencv** instalation with `pip`:

.. code-block:: bash

   $ pip uninstall opencv-python

Then install **opencv** from a linux repository using your current package
manager. For example, if you are using `apt` (for Debian/Ubuntu based systems)
you can install it by running:

.. code-block:: bash

   $ sudo apt install python3-opencv

.. note::

    The name of the package may vary. For example, in Arch repositories
    it is called **python-opencv**.

