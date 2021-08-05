Writting and Reading Trajectory objects
---------------------------------------

Regardless the source of the :py:class:`~yupi.Trajectory` object, you can save it on disk and later load it for further processing.

Writting Trajectory objects
+++++++++++++++++++++++++++

To store your :py:class:`~yupi.Trajectory` object, for instance, any of the ones created in the previous tutorials, you only need to call the :py:class:`~yupi.Trajectory.save` method as in:

.. code-block:: python

   track.save('spiral', file_type='json')


Reading Trajectory objects
++++++++++++++++++++++++++

To :py:class:`~yupi.Trajectory.load` a previously written :py:class:`~yupi.Trajectory` object:

.. code-block:: python

   from yupi import Trajectory
   track2 = Trajectory.load('spiral.json')

