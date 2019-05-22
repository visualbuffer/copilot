import pstats

p = pstats.Stats('profile.txt')
p.sort_stats('cumtime').print_stats(50)
