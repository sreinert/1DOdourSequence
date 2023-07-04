
from flip_tunnel_functions import make_flip_tunnel
from misc import default_main, save_yaml


options = default_main()
save_yaml(options)
no_input_speed = not ('speed' in options['inputs'])
flip_tunnel = make_flip_tunnel(
    options,
    test_mode=no_input_speed
)
flip_tunnel.run()
