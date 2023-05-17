
from flip_tunnel_functions import make_flip_tunnel
from misc import default_main


options = default_main()
no_input_speed = not ('speed' in options['inputs'])
flip_tunnel = make_flip_tunnel(
    options,
    test_mode=no_input_speed
)
flip_tunnel.run()
