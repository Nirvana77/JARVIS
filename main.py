# https://www.youtube.com/watch?v=pVZN03sNRxc
# https://www.youtube.com/watch?v=RAKpMYOlttA
# https://www.youtube.com/watch?v=OqFI_g8vAoc
# https://www.youtube.com/watch?v=1lwddP0KUEg

#import pandas as pd
import libs.openai_helper as openai
import libs.command_helper as commands

if __name__ == '__main__':
	print('Loading JARVIS...')
	
	openai.init()
	print('loading commands...')
	commands.init()
	

	commands.run()