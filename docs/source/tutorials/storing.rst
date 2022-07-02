Storage of Trajectory objects
-----------------------------

Regardless the source of the :py:class:`~trajectory.Trajectory` object, you can store it on disk and later load it for further processing or analysis.

Writing Trajectory objects
==========================

To store your :py:class:`~trajectory.Trajectory` object, (e.g. any of the ones created in the previous tutorials) you only need to call the `JSONSerializer` :py:func:`~core.JSONSerializer.save` method as in:

.. code-block:: python

   from yupi.core import JSONSerializer
   JSONSerializer.save(track, 'spiral.json')

This will produce a *json* file with all the details of the object that can be loaded anytime using yupi. Additionally, *csv* is another ``file_type`` available to store :py:class:`~trajectory.Trajectory` objects.


Reading Trajectory objects
==========================

To load a previously written :py:class:`~trajectory.Trajectory` object:

.. code-block:: python

   from yupi.core import JSONSerializer
   track2 = JSONSerializer.load('spiral.json')

