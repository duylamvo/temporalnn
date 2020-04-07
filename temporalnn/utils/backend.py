def tf_patch_rtx(mode='ignore'):
    """Fix-patch: Memory allocation in RTX cards - tensor-flow problem.
    Issues:
        [1]: Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
            https://github.com/tensorflow/tensorflow/issues/24496
    Usages:
        Run this script before build or fit data to train model.
    """
    try:
        import tensorflow as tf
        from keras import backend as K
        if hasattr(tf, "compat"):
            # version 2.0 tf
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            session = tf.compat.v1.Session(config=config)
            tf.compat.v1.keras.backend.set_session(session)
            print(tf.config.experimental.list_physical_devices('GPU'))
        else:
            # version 1
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            session = tf.Session(config=config)
            tf.keras.backend.set_session(session)

    except AttributeError as e:
        if mode == 'ignore':
            Warning(e)
        else:
            raise e
