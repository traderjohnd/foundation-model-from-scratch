# Notebook 06A final evidence package

The completed Notebook 06A evidence was packaged from persistent Google Drive after training, the one-time official-test evaluation, and the fixed D-091 generation probe were complete.

Five small canonical JSON artifacts are stored directly in this directory. The complete `extension_history.json` contains 9,600 optimizer-update records and is 3,151,019 bytes. Its canonical bytes were independently verified from the uploaded evidence ZIP against the packaging manifest:

- `extension_history.json` SHA-256: `cdc0cb63dfcdab65f1718347a4e352b2764a9d11cd6900144cdb9e6d279ed380`
- `extension_history.json` bytes: `3,151,019`
- uploaded six-file evidence ZIP SHA-256: `72efddf9586b1329a6d0676b44f4d9444211f8f4da6f2c9365772fbd3c25ef88`

The connected GitHub write interface used for repository closure accepts UTF-8 content but does not expose a direct local-file upload path for the multi-megabyte history JSON. Therefore the full history remains in the persistent Drive namespace and in the verified six-file evidence package; its content hash is committed in `artifact_manifest_sha256.json`. This is a transport limitation of the closure path, not a reconstruction or rounding of experimental evidence.

Large best/latest `.pt` checkpoints likewise remain external by design; their exact byte sizes and SHA-256 hashes are recorded in the manifest.
