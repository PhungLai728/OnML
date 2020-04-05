"""
Explanation class, with visualization functions.
"""
from __future__ import unicode_literals
from io import open
import os
import os.path
import json
import string
import numpy as np

# from .exceptions import LimeError
import exceptions
from exceptions import LimeError

from sklearn.utils import check_random_state


def text_separate2(idx):
    text_out = '''
    
    <table style="width:100%" >
          <tr class="lime top_div" id="top_div''' + str(idx) + '''">
            <td class="lime top_div1" id="top1_div''' + str(idx) + '''" style="width:50%; position: absolute; left: 0; top:0; padding-left: 5%"></td>
            <td class="lime top_div2" id="top2_div''' + str(idx) + '''" style="width:50%; position: absolute; right: 0; top:0;text-align: center"></td> 
          </tr>
          
          <tr class="lime header_div" id="header_div''' + str(idx) + '''">
            <th class="lime header_div1" id="header_div''' + str(idx) + '''" style="width:50%; position: absolute; right: 0; top: 200; text-align: center"><b>LIME</b></th>
            <th class="lime header_div3" id="header_div''' + str(idx) +  '''" style="width:50%; position: absolute; left: 0; top: 200; text-align: center"><b>OLLIE</b></th> 
            <th class="lime header_div2" id="header_div''' + str(idx) +  '''" style="width:50%; position: absolute; right: 0; top: 520; text-align: center"><b>Ontology</b></th> 
            <th class="lime header_div4" id="header_div''' + str(idx) +  '''" style="width:50%; position: absolute; left: 0; top: 520; text-align: center"><b>OnML</b></th> 
          </tr>

          <tr class="lime body_div" id="body_div''' + str(idx) + '''">
            <td class="lime body_div1" id="body1_div''' + str(idx) +  '''" style="width:60%; position: absolute; right: 0; top: 230; text-align: center"></td>
            <td class="lime body_div3" id="body3_div''' + str(idx) + '''" style="width:60%; position: absolute; left: 0; top: 230; text-align: center"></td> 
            <td class="lime body_div2" id="body2_div''' + str(idx) + '''" style="width:60%; position: absolute; right: 0; top: 550; text-align: center"></td> 
            <td class="lime body_div4" id="body4_div''' + str(idx) + '''" style="width:60%; position: absolute; left: 0; top: 550; text-align: center"></td> 
          </tr>
        </table>
        
        '''

    return text_out

def text_separate3(idx, rn ):
    if rn == 1:
        text_out = '''
    
        <table style="width:100%" >
          <tr class="lime top_div" id="top_div''' + str(idx) + '''">
            <td class="lime top_div1" id="top1_div''' + str(idx) + '''" style="width:50%; position: absolute; left: 0; top:0; padding-left: 5%"></td>
            <td class="lime top_div2" id="top2_div''' + str(idx) + '''" style="width:50%; position: absolute; right: 0; top:0;text-align: center"></td> 
          </tr>
          
          <tr class="lime header_div" id="header_div''' + str(idx) + '''">
            <th class="lime header_div1" id="header_div''' + str(idx) + '''" style="width:50%; position: absolute; left: 0; top: 250; text-align: center"><b>Algorithm 1</b></th>
            <th class="lime header_div2" id="header_div''' + str(idx) +  '''" style="width:50%; position: absolute; right: 0; top: 250; text-align: center"><b>Algorithm 2</b></th> 
            <th class="lime header_div3" id="header_div''' + str(idx) +  '''" style="width:50%; position: absolute; left: 0; top: 480; text-align: center"><b>Algorithm 3</b></th> 
            <th class="lime header_div4" id="header_div''' + str(idx) +  '''" style="width:50%; position: absolute; right: 0; top: 480; text-align: center"><b>Algorithm 4</b></th> 
          </tr>

          <tr class="lime body_div" id="body_div''' + str(idx) + '''">
            <td class="lime body_div1" id="body1_div''' + str(idx) +  '''" style="width:60%; position: absolute; left: 0; top: 280; text-align: center"></td>
            <td class="lime body_div2" id="body2_div''' + str(idx) + '''" style="width:60%; position: absolute; right: 0; top: 280; text-align: center"></td> 
            <td class="lime body_div3" id="body3_div''' + str(idx) + '''" style="width:60%; position: absolute; left: 0; top: 510; text-align: center"></td> 
            <td class="lime body_div4" id="body4_div''' + str(idx) + '''" style="width:60%; position: absolute; right: 0; top: 510; text-align: center"></td> 
          </tr>
        </table>
        
        '''
    elif rn == 2:
        text_out = '''
    
        <table style="width:100%" >
          <tr class="lime top_div" id="top_div''' + str(idx) + '''">
            <td class="lime top_div1" id="top1_div''' + str(idx) + '''" style="width:50%; position: absolute; left: 0; top:0; padding-left: 5%"></td>
            <td class="lime top_div2" id="top2_div''' + str(idx) + '''" style="width:50%; position: absolute; right: 0; top:0;text-align: center"></td> 
          </tr>
          
          <tr class="lime header_div" id="header_div''' + str(idx) + '''">
            <th class="lime header_div1" id="header_div''' + str(idx) + '''" style="width:50%; position: absolute; left: 0; top: 250; text-align: center"><b>Algorithm 1</b></th>
            <th class="lime header_div2" id="header_div''' + str(idx) +  '''" style="width:50%; position: absolute; right: 0; top: 250; text-align: center"><b>Algorithm 2</b></th> 
            <th class="lime header_div3" id="header_div''' + str(idx) +  '''" style="width:50%; position: absolute; left: 0; top: 480; text-align: center"><b>Algorithm 3</b></th> 
            <th class="lime header_div4" id="header_div''' + str(idx) +  '''" style="width:50%; position: absolute; right: 0; top: 480; text-align: center"><b>Algorithm 4</b></th> 
          </tr>

          <tr class="lime body_div" id="body_div''' + str(idx) + '''">
            <td class="lime body_div1" id="body1_div''' + str(idx) +  '''" style="width:60%; position: absolute; left: 0; top: 280; text-align: center"></td>
            <td class="lime body_div2" id="body2_div''' + str(idx) + '''" style="width:60%; position: absolute; right: 0; top: 280; text-align: center"></td> 
            <td class="lime body_div3" id="body3_div''' + str(idx) + '''" style="width:60%; position: absolute; right: 0; top: 510;  text-align: center"></td> 
            <td class="lime body_div4" id="body4_div''' + str(idx) + '''" style="width:60%; position: absolute; left: 0; top: 510; text-align: center"></td> 
          </tr>
        </table>
        
        '''
    elif rn == 3:
        text_out = '''
    
        <table style="width:100%" >
          <tr class="lime top_div" id="top_div''' + str(idx) + '''">
            <td class="lime top_div1" id="top1_div''' + str(idx) + '''" style="width:50%; position: absolute; left: 0; top:0; padding-left: 5%"></td>
            <td class="lime top_div2" id="top2_div''' + str(idx) + '''" style="width:50%; position: absolute; right: 0; top:0;text-align: center"></td> 
          </tr>
          
          <tr class="lime header_div" id="header_div''' + str(idx) + '''">
            <th class="lime header_div1" id="header_div''' + str(idx) + '''" style="width:50%; position: absolute; left: 0; top: 250; text-align: center"><b>Algorithm 1</b></th>
            <th class="lime header_div2" id="header_div''' + str(idx) +  '''" style="width:50%; position: absolute; right: 0; top: 250; text-align: center"><b>Algorithm 2</b></th> 
            <th class="lime header_div3" id="header_div''' + str(idx) +  '''" style="width:50%; position: absolute; left: 0; top: 480; text-align: center"><b>Algorithm 3</b></th> 
            <th class="lime header_div4" id="header_div''' + str(idx) +  '''" style="width:50%; position: absolute; right: 0; top: 480; text-align: center"><b>Algorithm 4</b></th> 
          </tr>

          <tr class="lime body_div" id="body_div''' + str(idx) + '''">
            <td class="lime body_div1" id="body1_div''' + str(idx) +  '''" style="width:60%; position: absolute; left: 0; top: 280; text-align: center"></td>
            <td class="lime body_div2" id="body2_div''' + str(idx) + '''" style="width:60%; position: absolute; left: 0; top: 510;  text-align: center"></td> 
            <td class="lime body_div3" id="body3_div''' + str(idx) + '''" style="width:60%; position: absolute; right: 0; top: 280; text-align: center"></td> 
            <td class="lime body_div4" id="body4_div''' + str(idx) + '''" style="width:60%; position: absolute; right: 0; top: 510; text-align: center"></td> 
          </tr>
        </table>
        
        '''
    elif rn == 4:
        text_out = '''
    
    <table style="width:100%" >
          <tr class="lime top_div" id="top_div''' + str(idx) + '''">
            <td class="lime top_div1" id="top1_div''' + str(idx) + '''" style="width:50%; position: absolute; left: 0; top:0; padding-left: 5%"></td>
            <td class="lime top_div2" id="top2_div''' + str(idx) + '''" style="width:50%; position: absolute; right: 0; top:0;text-align: center"></td> 
          </tr>
          
          <tr class="lime header_div" id="header_div''' + str(idx) + '''">
            <th class="lime header_div1" id="header_div''' + str(idx) + '''" style="width:50%; position: absolute; left: 0; top: 250; text-align: center"><b>Algorithm 1</b></th>
            <th class="lime header_div2" id="header_div''' + str(idx) +  '''" style="width:50%; position: absolute; right: 0; top: 250; text-align: center"><b>Algorithm 2</b></th> 
            <th class="lime header_div3" id="header_div''' + str(idx) +  '''" style="width:50%; position: absolute; left: 0; top: 480; text-align: center"><b>Algorithm 3</b></th> 
            <th class="lime header_div4" id="header_div''' + str(idx) +  '''" style="width:50%; position: absolute; right: 0; top: 480; text-align: center"><b>Algorithm 4</b></th> 
          </tr>

          <tr class="lime body_div" id="body_div''' + str(idx) + '''">
            <td class="lime body_div1" id="body1_div''' + str(idx) +  '''" style="width:60%; position: absolute; left: 0; top: 280; text-align: center"></td>
            <td class="lime body_div2" id="body2_div''' + str(idx) + '''" style="width:60%; position: absolute; right: 0; top: 510;   text-align: center"></td> 
            <td class="lime body_div3" id="body3_div''' + str(idx) + '''" style="width:60%; position: absolute; right: 0; top: 280;   text-align: center"></td> 
            <td class="lime body_div4" id="body4_div''' + str(idx) + '''" style="width:60%; position: absolute; left: 0; top: 510; text-align: center"></td> 
          </tr>
        </table>
        
        '''
    elif rn == 5:
        text_out = '''
    
        <table style="width:100%" >
          <tr class="lime top_div" id="top_div''' + str(idx) + '''">
            <td class="lime top_div1" id="top1_div''' + str(idx) + '''" style="width:50%; position: absolute; left: 0; top:0; padding-left: 5%"></td>
            <td class="lime top_div2" id="top2_div''' + str(idx) + '''" style="width:50%; position: absolute; right: 0; top:0;text-align: center"></td> 
          </tr>
          
          <tr class="lime header_div" id="header_div''' + str(idx) + '''">
            <th class="lime header_div1" id="header_div''' + str(idx) + '''" style="width:50%; position: absolute; left: 0; top: 250; text-align: center"><b>Algorithm 1</b></th>
            <th class="lime header_div2" id="header_div''' + str(idx) +  '''" style="width:50%; position: absolute; right: 0; top: 250; text-align: center"><b>Algorithm 2</b></th> 
            <th class="lime header_div3" id="header_div''' + str(idx) +  '''" style="width:50%; position: absolute; left: 0; top: 480; text-align: center"><b>Algorithm 3</b></th> 
            <th class="lime header_div4" id="header_div''' + str(idx) +  '''" style="width:50%; position: absolute; right: 0; top: 480; text-align: center"><b>Algorithm 4</b></th> 
          </tr>

          <tr class="lime body_div" id="body_div''' + str(idx) + '''">
            <td class="lime body_div1" id="body1_div''' + str(idx) +  '''" style="width:60%; position: absolute; left: 0; top: 280; text-align: center"></td>
            <td class="lime body_div2" id="body2_div''' + str(idx) + '''" style="width:60%; position: absolute; right: 0; top: 510; text-align: center"></td> 
            <td class="lime body_div3" id="body3_div''' + str(idx) + '''" style="width:60%; position: absolute; right: 0; top: 280; text-align: center"></td> 
            <td class="lime body_div4" id="body4_div''' + str(idx) + '''" style="width:60%; position: absolute;   left: 0; top: 510; text-align: center"></td> 
          </tr>
        </table>
        
        '''
    elif rn == 6:
        text_out = '''
    
        <table style="width:100%" >
          <tr class="lime top_div" id="top_div''' + str(idx) + '''">
            <td class="lime top_div1" id="top1_div''' + str(idx) + '''" style="width:50%; position: absolute; left: 0; top:0; padding-left: 5%"></td>
            <td class="lime top_div2" id="top2_div''' + str(idx) + '''" style="width:50%; position: absolute; right: 0; top:0;text-align: center"></td> 
          </tr>
          
          <tr class="lime header_div" id="header_div''' + str(idx) + '''">
            <th class="lime header_div1" id="header_div''' + str(idx) + '''" style="width:50%; position: absolute; left: 0; top: 250; text-align: center"><b>Algorithm 1</b></th>
            <th class="lime header_div2" id="header_div''' + str(idx) +  '''" style="width:50%; position: absolute; right: 0; top: 250; text-align: center"><b>Algorithm 2</b></th> 
            <th class="lime header_div3" id="header_div''' + str(idx) +  '''" style="width:50%; position: absolute; left: 0; top: 480; text-align: center"><b>Algorithm 3</b></th> 
            <th class="lime header_div4" id="header_div''' + str(idx) +  '''" style="width:50%; position: absolute; right: 0; top: 480; text-align: center"><b>Algorithm 4</b></th> 
          </tr>

          <tr class="lime body_div" id="body_div''' + str(idx) + '''">
            <td class="lime body_div1" id="body1_div''' + str(idx) +  '''" style="width:60%; position: absolute; left: 0; top: 280; text-align: center"></td>
            <td class="lime body_div2" id="body2_div''' + str(idx) + '''" style="width:60%; position: absolute; right: 0; top: 510;  text-align: center"></td> 
            <td class="lime body_div3" id="body3_div''' + str(idx) + '''" style="width:60%; position: absolute; left: 0; top: 510; text-align: center"></td> 
            <td class="lime body_div4" id="body4_div''' + str(idx) + '''" style="width:60%; position: absolute; right: 0; top: 280; text-align: center"></td> 
          </tr>
        </table>
        
        '''
    elif rn == 7:
        text_out = '''
    
        <table style="width:100%" >
          <tr class="lime top_div" id="top_div''' + str(idx) + '''">
            <td class="lime top_div1" id="top1_div''' + str(idx) + '''" style="width:50%; position: absolute; left: 0; top:0; padding-left: 5%"></td>
            <td class="lime top_div2" id="top2_div''' + str(idx) + '''" style="width:50%; position: absolute; right: 0; top:0;text-align: center"></td> 
          </tr>
          
          <tr class="lime header_div" id="header_div''' + str(idx) + '''">
            <th class="lime header_div1" id="header_div''' + str(idx) + '''" style="width:50%; position: absolute; left: 0; top: 250; text-align: center"><b>Algorithm 1</b></th>
            <th class="lime header_div2" id="header_div''' + str(idx) +  '''" style="width:50%; position: absolute; right: 0; top: 250; text-align: center"><b>Algorithm 2</b></th> 
            <th class="lime header_div3" id="header_div''' + str(idx) +  '''" style="width:50%; position: absolute; left: 0; top: 480; text-align: center"><b>Algorithm 3</b></th> 
            <th class="lime header_div4" id="header_div''' + str(idx) +  '''" style="width:50%; position: absolute; right: 0; top: 480; text-align: center"><b>Algorithm 4</b></th> 
          </tr>

          <tr class="lime body_div" id="body_div''' + str(idx) + '''">
            <td class="lime body_div1" id="body1_div''' + str(idx) +  '''" style="width:60%; position: absolute; right: 0; top: 280;   text-align: center"></td>
            <td class="lime body_div2" id="body2_div''' + str(idx) + '''" style="width:60%; position: absolute; left: 0; top: 280; text-align: center"></td> 
            <td class="lime body_div3" id="body3_div''' + str(idx) + '''" style="width:60%; position: absolute; left: 0; top: 510; text-align: center"></td> 
            <td class="lime body_div4" id="body4_div''' + str(idx) + '''" style="width:60%; position: absolute; right: 0; top: 510; text-align: center"></td> 
          </tr>
        </table>
        
        '''
    elif rn == 8:
        text_out = '''
    
        <table style="width:100%" >
          <tr class="lime top_div" id="top_div''' + str(idx) + '''">
            <td class="lime top_div1" id="top1_div''' + str(idx) + '''" style="width:50%; position: absolute; left: 0; top:0; padding-left: 5%"></td>
            <td class="lime top_div2" id="top2_div''' + str(idx) + '''" style="width:50%; position: absolute; right: 0; top:0;text-align: center"></td> 
          </tr>
          
          <tr class="lime header_div" id="header_div''' + str(idx) + '''">
            <th class="lime header_div1" id="header_div''' + str(idx) + '''" style="width:50%; position: absolute; left: 0; top: 250; text-align: center"><b>Algorithm 1</b></th>
            <th class="lime header_div2" id="header_div''' + str(idx) +  '''" style="width:50%; position: absolute; right: 0; top: 250; text-align: center"><b>Algorithm 2</b></th> 
            <th class="lime header_div3" id="header_div''' + str(idx) +  '''" style="width:50%; position: absolute; left: 0; top: 480; text-align: center"><b>Algorithm 3</b></th> 
            <th class="lime header_div4" id="header_div''' + str(idx) +  '''" style="width:50%; position: absolute; right: 0; top: 480; text-align: center"><b>Algorithm 4</b></th> 
          </tr>

          <tr class="lime body_div" id="body_div''' + str(idx) + '''"> 
            <td class="lime body_div1" id="body1_div''' + str(idx) +  '''" style="width:60%; position: absolute; right: 0; top: 280;  text-align: center"></td>
            <td class="lime body_div2" id="body2_div''' + str(idx) + '''" style="width:60%; position: absolute; left: 0; top: 280; text-align: center"></td> 
            <td class="lime body_div3" id="body3_div''' + str(idx) + '''" style="width:60%; position: absolute; right: 0; top: 510;  text-align: center"></td> 
            <td class="lime body_div4" id="body4_div''' + str(idx) + '''" style="width:60%; position: absolute; left: 0; top: 510;text-align: center"></td> 
          </tr>
        </table>
        
        '''
    # elif rn == 9:
    #     pos = [alg2,alg3,alg1,alg4]
    #     z = [2,3,1,4]
    # elif rn == 10:
    #     pos = [alg2,alg3,alg4,alg1]
    #     z = [2,3,4,1]
    # elif rn == 11:
    #     pos = [alg2,alg4,alg1,alg3]
    #     z = [2,4,1,3]
    # elif rn == 12:
    #     pos = [alg2,alg4,alg3,alg1]
    #     z = [2,4,3,1]
    # elif rn == 13:
    #     pos = [alg3,alg1,alg2,alg4]
    #     z = [3,1,2,4]
    # elif rn == 14:
    #     pos = [alg3,alg1,alg4,alg2]
    #     z = [3,1,4,2]
    # elif rn == 15:
    #     pos = [alg3,alg2,alg1,alg4]
    #     z = [3,2,1,4]
    # elif rn == 16:
    #     pos = [alg3,alg2,alg4,alg1]
    #     z = [3,2,4,1]
    # elif rn == 17:
    #     pos = [alg3,alg4,alg1,alg2]
    #     z = [3,4,1,2]
    # elif rn == 18:
    #     pos = [alg3,alg4,alg2,alg1]
    #     z = [3,4,2,1]
    # elif rn == 19:
    #     pos = [alg4,alg1,alg2,alg3]
    #     z = [4,1,2,3]
    # elif rn == 20:
    #     pos = [alg4,alg1,alg3,alg2]
    #     z = [4,1,3,2]
    # elif rn == 21:
    #     pos = [alg4,alg2,alg1,alg3]
    #     z = [4,2,1,3]
    # elif rn == 22:
    #     pos = [alg4,alg2,alg3,alg1]
    #     z = [4,2,3,1]
    # elif rn == 23:
    #     pos = [alg4,alg3,alg1,alg2]
    #     z = [4,3,1,2]
    else:
        print('ok')
        # pos = [alg4,alg3,alg2,alg1]
        # z = [4,3,2,1]

   

    return text_out

# def text_separate(idx, rn ):
#     text_out = '''
    
#     <table style="width:100%" >
#           <tr class="lime top_div" id="top_div''' + str(idx) + '''">
#             <td class="lime top_div1" id="top1_div''' + str(idx) + '''" style="width:50%; position: absolute; left: 0; top:0; padding-left: 5%"></td>
#             <td class="lime top_div2" id="top2_div''' + str(idx) + '''" style="width:50%; position: absolute; right: 0; top:0;text-align: center"></td> 
#           </tr>
          
#           <tr class="lime header_div" id="header_div''' + str(idx) + '''">
#             <th class="lime header_div1" id="header_div''' + str(idx) + '''" style="width:50%; position: absolute; left: 0; top: 250; text-align: center"><b>Algorithm 1</b></th>
#             <th class="lime header_div3" id="header_div''' + str(idx) +  '''" style="width:50%; position: absolute; right: 0; top: 250; text-align: center"><b>Algorithm 2</b></th> 
#             <th class="lime header_div2" id="header_div''' + str(idx) +  '''" style="width:50%; position: absolute; left: 0; top: 480; text-align: center"><b>Algorithm 3</b></th> 
#             <th class="lime header_div4" id="header_div''' + str(idx) +  '''" style="width:50%; position: absolute; right: 0; top: 480; text-align: center"><b>Algorithm 4</b></th> 
#           </tr>

#           <tr class="lime body_div" id="body_div''' + str(idx) + '''">
#             <td class="lime body_div1" id="body1_div''' + str(idx) +  '''" style="width:60%; position: absolute; left: 0; top: 280; text-align: center"></td>
#             <td class="lime body_div3" id="body3_div''' + str(idx) + '''" style="width:60%; position: absolute; right: 0; top: 280; text-align: center"></td> 
#             <td class="lime body_div2" id="body2_div''' + str(idx) + '''" style="width:60%; position: absolute; left: 0; top: 510; text-align: center"></td> 
#             <td class="lime body_div4" id="body4_div''' + str(idx) + '''" style="width:60%; position: absolute; right: 0; top: 510; text-align: center"></td> 
#           </tr>
#         </table>
        
#         '''

#     return text_out

def text_separate(idx, rn ):
    alg1 = ['left: 0; top: 250; text-align: center"><b>Algorithm 1', 'left: 0; top: 280;' ]
    alg2 = ['right: 0; top: 250;  text-align: center"><b>Algorithm 2', 'right: 0; top: 280;' ]
    alg3 = ['left: 0; top: 480; text-align: center"><b>Algorithm 3', 'left: 0; top: 510;']
    alg4 = ['right: 0; top: 480; text-align: center"><b>Algorithm 4', 'right: 0; top: 510;']
    if rn == 1:
        pos = [alg1,alg2,alg3,alg4]
        z = [1,2,3,4]
    elif rn == 2:
        pos = [alg1,alg2,alg4,alg3]
        z = [1,2,4,3]
    elif rn == 3:
        pos = [alg1,alg3,alg2,alg4]
        z = [1,3,2,4]
    elif rn == 4: #5
        pos = [alg1,alg4,alg2,alg3]
        z = [1,4,2,3]
    elif rn == 5: #4
        pos = [alg1,alg3,alg4,alg2]
        z = [1,3,4,2]
    elif rn == 6:
        pos = [alg1,alg4,alg3,alg2]
        z = [1,4,3,2]
    elif rn == 7:
        pos = [alg2,alg1,alg3,alg4]
        z = [2,1,3,4]
    elif rn == 8:
        pos = [alg2,alg1,alg4,alg3]
        z = [2,1,4,3]
    elif rn == 9: #13
        pos = [alg3,alg1,alg2,alg4]
        z = [3,1,2,4]
    elif rn == 10: #19
        pos = [alg4,alg1,alg2,alg3]
        z = [4,1,2,3]
    elif rn == 11: #14
        pos = [alg3,alg1,alg4,alg2]
        z = [3,1,4,2]
    elif rn == 12: #20
        pos = [alg4,alg1,alg3,alg2]
        z = [4,1,3,2]
    elif rn == 13: #9
        pos = [alg2,alg3,alg1,alg4]
        z = [2,3,1,4]
    elif rn == 14: #11
        pos = [alg2,alg4,alg1,alg3]
        z = [2,4,1,3]
    elif rn == 15:
        pos = [alg3,alg2,alg1,alg4]
        z = [3,2,1,4]
    elif rn == 16: #21
        pos = [alg4,alg2,alg1,alg3]
        z = [4,2,1,3]
    elif rn == 17:
        pos = [alg3,alg4,alg1,alg2]
        z = [3,4,1,2]
    elif rn == 18: #23
        pos = [alg4,alg3,alg1,alg2]
        z = [4,3,1,2]
    elif rn == 19: #10
        pos = [alg2,alg3,alg4,alg1]
        z = [2,3,4,1]
    elif rn == 20: #12
        pos = [alg2,alg4,alg3,alg1]
        z = [2,4,3,1]
    elif rn == 21: #16
        pos = [alg3,alg2,alg4,alg1]
        z = [3,2,4,1]
    elif rn == 22:
        pos = [alg4,alg2,alg3,alg1]
        z = [4,2,3,1]
    elif rn == 23: #18
        pos = [alg3,alg4,alg2,alg1]
        z = [3,4,2,1]
    else:
        pos = [alg4,alg3,alg2,alg1]
        z = [4,3,2,1]

    x = [pos[0][0], pos[1][0], pos[2][0], pos[3][0]]
    y_edit = [pos[0][1], pos[1][1], pos[2][1], pos[3][1]]


    # y_edit = []
    # for k in z:
    #     y_edit.append(y[k-1])
    
    text_out = '''
    
    <table style="width:100%" >
          <tr class="lime top_div" id="top_div''' + str(idx) + '''">
            <td class="lime top_div1" id="top1_div''' + str(idx) + '''" style="width:50%; position: absolute; left: 0; top:0; padding-left: 5%"></td>
            <td class="lime top_div2" id="top2_div''' + str(idx) + '''" style="width:50%; position: absolute; right: 0; top:0;text-align: center"></td> 
          </tr>
          
          <tr class="lime header_div" id="header_div''' + str(idx) + '''">
            <th class="lime header_div1" id="header_div''' + str(idx) + '''" style="width:50%; position: absolute; ''' + x[0] +  '''</b></th>
            <th class="lime header_div2" id="header_div''' + str(idx) +  '''" style="width:50%; position: absolute; ''' + x[1] +  '''</b></th> 
            <th class="lime header_div3" id="header_div''' + str(idx) +  '''" style="width:50%; position: absolute; ''' + x[2] +  '''</b></th> 
            <th class="lime header_div4" id="header_div''' + str(idx) +  '''" style="width:50%; position: absolute; ''' + x[3] +  '''</b></th> 
          </tr>

          <tr class="lime body_div" id="body_div''' + str(idx) + '''">
            <td class="lime body_div1" id="body1_div''' + str(idx) +  '''" style="width:60%; position: absolute; ''' + y_edit[0] +  ''' text-align: center"></td>
            <td class="lime body_div2" id="body2_div''' + str(idx) + '''" style="width:60%; position: absolute; ''' + y_edit[1] +  ''' text-align: center"></td> 
            <td class="lime body_div3" id="body3_div''' + str(idx) + '''" style="width:60%; position: absolute; ''' + y_edit[2] +  ''' text-align: center"></td> 
            <td class="lime body_div4" id="body4_div''' + str(idx) + '''" style="width:60%; position: absolute; ''' + y_edit[3] +  ''' text-align: center"></td> 
          </tr>
        </table>
        
        '''

    return text_out


def id_generator(size=15, random_state=None):
    """Helper function to generate random div ids. This is useful for embedding
    HTML into ipython notebooks."""
    chars = list(string.ascii_uppercase + string.digits)
    return ''.join(random_state.choice(chars, size, replace=True))


class DomainMapper(object):
    """Class for mapping features to the specific domain.

    The idea is that there would be a subclass for each domain (text, tables,
    images, etc), so that we can have a general Explanation class, and separate
    out the specifics of visualizing features in here.
    """

    def __init__(self):
        pass

    def map_exp_ids(self, exp, **kwargs):
        """Maps the feature ids to concrete names.

        Default behaviour is the identity function. Subclasses can implement
        this as they see fit.

        Args:
            exp: list of tuples [(id, weight), (id,weight)]
            kwargs: optional keyword arguments

        Returns:
            exp: list of tuples [(name, weight), (name, weight)...]
        """
        return exp

    def visualize_instance_html(self,
                                exp,
                                label,
                                div_name,
                                iss_name,
                                exp_object_name,
                                **kwargs):
        """Produces html for visualizing the instance.

        Default behaviour does nothing. Subclasses can implement this as they
        see fit.

        Args:
             exp: list of tuples [(id, weight), (id,weight)]
             label: label id (integer)
             div_name: name of div object to be used for rendering(in js)
             exp_object_name: name of js explanation object
             kwargs: optional keyword arguments

        Returns:
             js code for visualizing the instance
        """
        return ''


class Explanation(object):
    """Object returned by explainers."""

    def __init__(self,
                 domain_mapper,
                 mode='classification',
                 class_names=None,
                 random_state=None):
        """

        Initializer.

        Args:
            domain_mapper: must inherit from DomainMapper class
            type: "classification" or "regression"
            class_names: list of class names (only used for classification)
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        self.random_state = random_state
        self.mode = mode
        self.domain_mapper = domain_mapper
        self.local_exp = {}
        self.intercept = {}
        self.score = None
        self.local_pred = None
        self.scaled_data = None
        if mode == 'classification':
            self.class_names = class_names
            self.top_labels = None
            self.predict_proba = None
        elif mode == 'regression':
            self.class_names = ['negative', 'positive']
            self.predicted_value = None
            self.min_value = 0.0
            self.max_value = 1.0
            self.dummy_label = 1
        else:
            raise LimeError('Invalid explanation mode "{}". '
                            'Should be either "classification" '
                            'or "regression".'.format(mode))

    def available_labels(self):
        """
        Returns the list of classification labels for which we have any explanations.
        """
        try:
            assert self.mode == "classification"
        except AssertionError:
            raise NotImplementedError('Not supported for regression explanations.')
        else:
            ans = self.top_labels if self.top_labels else self.local_exp.keys()
            return list(ans)

    def as_list(self, label=1, **kwargs):
        """Returns the explanation as a list.

        Args:
            label: desired label. If you ask for a label for which an
                explanation wasn't computed, will throw an exception.
                Will be ignored for regression explanations.
            kwargs: keyword arguments, passed to domain_mapper

        Returns:
            list of tuples (representation, weight), where representation is
            given by domain_mapper. Weight is a float.
        """
        label_to_use = label if self.mode == "classification" else self.dummy_label
        ans = self.domain_mapper.map_exp_ids(self.local_exp[label_to_use], **kwargs)

        return ans

    def as_map(self):
        """Returns the map of explanations.

        Returns:
            Map from label to list of tuples (feature_id, weight).
        """
        return self.local_exp

    def as_pyplot_figure(self, label=1, **kwargs):
        """Returns the explanation as a pyplot figure.

        Will throw an error if you don't have matplotlib installed
        Args:
            label: desired label. If you ask for a label for which an
                   explanation wasn't computed, will throw an exception.
                   Will be ignored for regression explanations.
            kwargs: keyword arguments, passed to domain_mapper

        Returns:
            pyplot figure (barchart).
        """
        import matplotlib.pyplot as plt
        exp = self.as_list(label=label, **kwargs)
        fig = plt.figure()
        vals = [x[1] for x in exp]
        names = [x[0] for x in exp]
        vals.reverse()
        names.reverse()
        colors = ['green' if x > 0 else 'red' for x in vals]
        pos = np.arange(len(exp)) + .5
        plt.barh(pos, vals, align='center', color=colors)
        plt.yticks(pos, names)
        if self.mode == "classification":
            title = 'Local explanation for class %s' % self.class_names[label]
        else:
            title = 'Local explanation'
        plt.title(title)
        return fig

    def show_in_notebook(self,
                         labels=None,
                         predict_proba=True,
                         show_predicted_value=True,
                         **kwargs):
        """Shows html explanation in ipython notebook.

        See as_html() for parameters.
        This will throw an error if you don't have IPython installed"""

        from IPython.core.display import display, HTML
        display(HTML(self.as_html(labels=labels,
                                  predict_proba=predict_proba,
                                  show_predicted_value=show_predicted_value,
                                  **kwargs)))

    def save_to_file(self,
                     file_path,
                     new_rule,
                     OLLIE_rule,
                     ensemble_rule,
                     iss_name,
                     rn,
                     labels=None,
                     predict_proba=True,
                     show_predicted_value=True,
                     **kwargs):
        """Saves html explanation to file. .

        Params:
            file_path: file to save explanations to

        See as_html() for additional parameters.

        """
        file_ = open(file_path, 'w', encoding='utf8')
        file_.write(self.as_html(labels=labels,
                                 predict_proba=predict_proba,
                                 new_rule=new_rule,OLLIE_rule=OLLIE_rule,ensemble_rule=ensemble_rule,iss_name=iss_name,rn=rn,
                                 show_predicted_value=show_predicted_value,
                                 **kwargs))
        file_.close()

    

    def as_html(self,
                labels=None,
                predict_proba=True,
                new_rule=True,
                OLLIE_rule=True,
                ensemble_rule=True,
                iss_name=True,
                rn = True,
                show_predicted_value=True,
                **kwargs):
        """Returns the explanation as an html page.

        Args:
            labels: desired labels to show explanations for (as barcharts).
                If you ask for a label for which an explanation wasn't
                computed, will throw an exception. If None, will show
                explanations for all available labels. (only used for classification)
            predict_proba: if true, add  barchart with prediction probabilities
                for the top classes. (only used for classification)
            show_predicted_value: if true, add  barchart with expected value
                (only used for regression)
            kwargs: keyword arguments, passed to domain_mapper

        Returns:
            code for an html page, including javascript includes.
        """

        def jsonize(x):
            return json.dumps(x, ensure_ascii=False)

        if labels is None and self.mode == "classification":
            labels = self.available_labels()

        this_dir, _ = os.path.split(__file__)
        bundle = open(os.path.join(this_dir, 'bundle_cc.js'),
                      encoding="utf8").read()

        out = u'''<html>
        <meta http-equiv="content-type" content="text/html; charset=UTF8">
        <head><script>%s </script></head><body>''' % bundle
        random_id = id_generator(size=15, random_state=check_random_state(self.random_state))

        out += text_separate2(random_id)
        # out += text_separate(random_id, rn)


        raw_js = '''var top_div1 = d3.select('#top1_div%s').classed('lime top_div1', true);
            var pp_div = top_div1.append('div')
                                .classed('lime predict_proba', true);
            var exp = new lime.Explanation([""]);
            var raw_div = pp_div.append('div');
            ''' % (random_id)
        html_data = []
        add = self.domain_mapper.visualize_instance_html(
                html_data,
                labels[0] if self.mode == "classification" else self.dummy_label,
                'raw_div',
                'exp',
                **kwargs)
        add_tk = add.split()
        add1 = '\n      ' + ' '.join([add_tk[i] for i in range(len(add_tk)-1)])
        add2 = ''' "%s" ''' % (iss_name)
        add3 =  ', ' + add_tk[-1][:4]  + ');  \n'

        to_add = add1 + ' ' + add2 + add3 
        raw_js += to_add

        # raw_js += self.domain_mapper.visualize_instance_html(
        #         html_data,
        #         labels[0] if self.mode == "classification" else self.dummy_label,
        #         'raw_div',
        #         'exp',
        #         **kwargs)

# def visualize_instance_html(self, exp, label, div_name, iss_name, exp_object_name,
#                                 text=True, opacity=True):

        predict_proba_js = ''
        if self.mode == "classification" and predict_proba:
            predict_proba_js = u'''
            var top_div2 = d3.select('#top2_div%s').classed('lime top_div2', true);
            var pp_div2 = top_div2.append('div')
                                .classed('lime predict_proba', true);
            var pp_svg = pp_div2.append('svg');
            var pp = new lime.PredictProba(pp_svg, %s, %s);
            ''' % (random_id,jsonize(self.class_names),
                   jsonize(list(self.predict_proba.astype(float))))
            # ''' % (random_id)

        exp_js_lime = '''var body_div1 = d3.select('#body1_div%s').classed('lime body_div1', true); 
            var exp_div;
            var exp = new lime.Explanation(["0", "1"]);
        ''' % (random_id)

        if self.mode == "classification":
            for label in labels:
                exp = jsonize(self.as_list(label))
                exp_js_lime += u'''
                exp_div = body_div1.append('div').classed('lime explanation', true);
                exp.show(%s, %d, exp_div);
                ''' % (exp, label)
        else:
            exp = jsonize(self.as_list())
            exp_js_lime += u'''
            exp_div = body_div1.append('div').classed('lime explanation', true);
            exp.show(%s, %s, exp_div);
            ''' % (exp, self.dummy_label)

        exp_js_ours = '''var body_div2 = d3.select('#body2_div%s').classed('lime body_div2', true); 
            var exp_div2;
            var exp = new lime.Explanation(["0", "1"]);
        ''' % (random_id)

        if self.mode == "classification":
            for label in labels:
                exp = jsonize(self.as_list(label))
                exp_js_ours += u'''
                exp_div2 = body_div2.append('div').classed('lime explanation', true);
                exp.show(%s, %d, exp_div2);
                ''' % (new_rule, label)

        exp_js_ollie = '''var body_div3 = d3.select('#body3_div%s').classed('lime body_div3', true); 
            var exp_div3;
            var exp = new lime.Explanation(["0", "1"]);
        ''' % (random_id)

        if self.mode == "classification":
            for label in labels:
                exp = jsonize(self.as_list(label))
                exp_js_ollie += u'''
                exp_div3 = body_div3.append('div').classed('lime explanation', true);
                exp.show(%s, %d, exp_div3);
                ''' % (OLLIE_rule, label)
        
        exp_js_ensemble = '''var body_div4 = d3.select('#body4_div%s').classed('lime body_div4', true); 
            var exp_div4;
            var exp = new lime.Explanation(["0", "1"]);
        ''' % (random_id)

        if self.mode == "classification":
            for label in labels:
                exp = jsonize(self.as_list(label))
                exp_js_ensemble += u'''
                exp_div4 = body_div4.append('div').classed('lime explanation', true);
                exp.show(%s, %d, exp_div4);
                ''' % (ensemble_rule, label)


        out += u'''

        <script>
        %s
        %s
        %s
        %s
        %s
        %s

        </script>
        ''' % (raw_js, predict_proba_js, exp_js_lime, exp_js_ours, exp_js_ollie, exp_js_ensemble)
        out += u'</body></html>'

        return out
