# Classification resources

## Danish SKS hierarchy

`../ehr2meds/resources/sks_hierarchy.parquet` is derived from the official `SKScomplete.txt`
exchange file published by Sundhedsdatastyrelsen:

<https://filer.sundhedsdata.dk/sks/data/skscomplete/SKScomplete.txt>

The snapshot was downloaded on 2026-07-28. Its source SHA-256 is:

```text
27cbf709205ad087bbbbad62a854afce579074d0007c5d7b1fe83e3b8a2f35cd
```

The resource retains the official code versions, validity intervals,
classification record types, descriptions, and selected validation fields.
`parent_code` is the nearest shorter prefix that is also present as an official
code anywhere in the SKS catalogue. It is null when no official prefix ancestor
exists. In particular, the resource does not manufacture parents for flat SKS
branches.

Regenerate it from a downloaded source file with:

```shell
python ehr2meds/build_sks_hierarchy.py \
  --source /path/to/SKScomplete.txt \
  --output ehr2meds/resources/sks_hierarchy.parquet
```

Or download and build it in one command:

```shell
python ehr2meds/build_sks_hierarchy.py \
  --download-to /tmp/SKScomplete.txt \
  --output ehr2meds/resources/sks_hierarchy.parquet
```

Project-specific corrections and synthetic hierarchy nodes should be stored in
a separate external mapping. They must not be edited into this generated base,
because regeneration would overwrite them.

The source classification is maintained and copyrighted by
Sundhedsdatastyrelsen. Review its applicable reuse terms before redistributing
the derived resource outside this project.
