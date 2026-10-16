{#
  Document only the members defined in the class itself, in source order: the
  public classes are Pydantic models, and listing the members inherited from
  ``pydantic.BaseModel`` (``model_dump``, ``model_validate``, etc.) would bury
  the settings fields.
#}
{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :member-order: bysource
   :exclude-members: model_config
