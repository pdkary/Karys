from models.progressive_discriminator import ProgressiveDiscriminator
from models.progressive_generator import ProgressiveGenerator

def test_build_prog_gan():
    disc = ProgressiveDiscriminator(4096)
    
    disc_graph = disc.build_graph()
    print(disc_graph.input_shape)
    print(disc_graph.output_shape)

    gen = ProgressiveGenerator(512)
    
    gen_graph = gen.build_graph()
    print(gen_graph.input_shape)
    print(gen_graph.output_shape)


