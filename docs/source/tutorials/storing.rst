Storage of Trajectory objects
-----------------------------

Regardless the source of the :py:class:`~yupi.Trajectory` object, you can store it on disk and later load it for further processing or analysis.

Writing Trajectory objects
==========================

To store your :py:class:`~yupi.Trajectory` object, (e.g. any of the ones created in the previous tutorials) you only need to call the :py:func:`~yupi.Trajectory.save` method as in:

.. code-block:: python

   track.save('spiral', file_type='json')

This will produce a *json* file with all the details of the object that can be loaded anytime using yupi. Additionally, *csv* is another ``file_type`` available to store :py:class:`~yupi.Trajectory` objects.


Reading Trajectory objects
==========================

To :py:func:`~yupi.Trajectory.load` a previously written :py:class:`~yupi.Trajectory` object:

.. code-block:: python

   from yupi import Trajectory
   track2 = Trajectory.load('spiral.json')

