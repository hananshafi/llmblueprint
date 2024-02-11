bg_prompt_text = "Background prompt: "

default_template = """"You are an intelligent bounding box generator. I will provide you with a caption for \
a photo, image, detailed scene or painting. Your task is to generate the bounding boxes for the objects mentioned in the caption, \
along with a background prompt describing the scene. The images are of size 512x512, and the bounding boxes \
must not overlap or go beyond the image boundaries. Each bounding box should be in the format of \ 
(object name, [top-left x coordinate, top-left y coordinate, box width, box height]) and \ 
include exactly one object. Make the boxes larger if possible. Keep the boxes as far apart from each other as possible. \
Do not put objects that are already provided in the bounding boxes into the background prompt. \ 
If needed, you can make reasonable guesses. Generate the object descriptions and background prompts \
in English even if the caption might not be in English. Do not include non-existing or excluded objects \
in the background prompt. Please refer to the examples below for the desired format.

Caption: A realistic image of landscape scene depicting a green car parking on the left of a blue truck, with a red air balloon and a bird in the sky
Objects: [('a green car', [21, 181, 211, 159]), ('a blue truck', [269, 181, 209, 160]), ('a red air balloon', [66, 8, 145, 135]), ('a bird', [296, 42, 143, 100])]
Background prompt: A realistic image of a landscape scene

Caption: A watercolor painting of a wooden table in the living room with an apple on it
Objects: [('a wooden table', [65, 243, 344, 206]), ('a apple', [206, 306, 81, 69])]
Background prompt: A watercolor painting of a living room

Caption: A watercolor painting of two pandas eating bamboo in a forest
Objects: [('a panda eating bambooo', [30, 171, 212, 226]), ('a panda eating bambooo', [264, 173, 222, 221])]
Background prompt: A watercolor painting of a forest

Caption: A realistic image of four skiers standing in a line on the snow near a palm tree
Objects: [('a skier', [5, 152, 139, 168]), ('a skier', [278, 192, 121, 158]), ('a skier', [148, 173, 124, 155]), ('a palm tree', [404, 180, 103, 180])]
Background prompt: A realistic image of an outdoor scene with snow

Caption: An oil painting of a pink dolphin jumping on the left of a steam boat on the sea
Objects: [('a steam boat', [232, 225, 257, 149]), ('a jumping pink dolphin', [21, 249, 189, 123])]
Background prompt: An oil painting of the sea

Caption: A realistic image of a cat playing with a dog in a park with flowers
Objects: [('a playful cat', [51, 67, 271, 324]), ('a playful dog', [302, 119, 211, 228])]
Background prompt: A realistic image of a park with flowers"""


#Also take into account living vs non-living criterion with lifeless objects preceeding the living ones. \
extract_obj_prompt = """You are an intelligent description extractor. \
                    I will give you a list of the objects and a corresponding text prompt. \                    
                    For each object, extract its respective description or details \
            mentioned in the text prompt. The description should strictly contain the fine details \
            about the object and must not contain information regarding location or abstract details \
            about the object. The description must also \
             contain the name of the object being described. For objects which do not have concrete \
             description mentioned, return the object itself in that case. The output should be a python dictionary \
            with key as object and value as description. The description should start with 'A realistic photo of \
            object' followed by its characteristics. Sort the entries as per objects which are spatially \
            behind (background) followed by objects which are spatially ahead (foreground). \
            For instance object "a garden view" should preceed the "table". Make an intelligent guess if possible.\
            Here are some examples:      
            list of objects: ['a Golden Retriever','a white cat','a wooden table','a vase of vibrant flowers',
                             'a sleek modern television']
            text_prompt: "In a cozy living room, a heartwarming scene unfolds. A friendly and affectionate \
            Golden Retriever with a soft, golden-furred coat rests contently on a plush rug, its warm eyes \
            filled with joy. Nearby, a graceful and elegant white cat stretches leisurely, showcasing its \
            pristine and fluffy fur. A sturdy wooden table with polished edges stands gracefully in the \
            center, adorned with a vase of vibrant flowers adding a touch of freshness. On the wall, \
            a sleek modern television stands ready to provide entertainment. The ambiance is warm, inviting, \
            and filled with a sense of companionship and relaxation."
            output: {'a sleek modern television': 'A realistic photo of a sleek modern television stands ready to provide entertainment.' \
            'a wooden table': 'A realistic photo of a sturdy wooden table with polished edges.', \
        'a vase of vibrant flowers': 'A realistic photo of a vase of vibrant flowers adding a touch of freshness.', \             
            'a Golden Retriever': 'A realistic photo of a friendly and affectionate Golden Retriever with a \
            soft, golden-furred coat and its warm eyes filled with joy.',\
            'a white cat': 'A realistic photo of a graceful and elegant white cat stretches leisurely, showcasing its pristine and \
            fluffy fur.'}
           
            list of objects: ['a red farmhouse','a weathered picket fence', 'an antique tractor','a scarecrow',
                            'an antique tractor']
            text_prompt: "In the quiet countryside, a red farmhouse stands with an old-fashioned charm. \
            Nearby, a weathered picket fence surrounds a garden of wildflowers. An antique tractor, though worn, \
            rests as a reminder of hard work. A scarecrow watches over fields of swaying crops. The air carries \
            the scent of earth and hay. Set against rolling hills, this farmhouse tells a story of connection to \
            the land and its traditions."
            output: {'a red farmhouse': 'A realistic photo of a red farmhouse with an \
            old-fashioned charm in the quiet countryside.', 'a weathered picket fence': 'A realistic photo of a\
            weathered picket fence surrounding a garden of wildflowers.','an antique tractor': 'A realistic photo of \
            an antique tractor, though worn, rests as a reminder of hard work.',
            'a scarecrow': 'A realistic photo of a scarecrow watching over fields of swaying crops.'}
            """

simplified_prompt = """{template}

Caption: {prompt}
Objects: """

prompt_placeholder = "A realistic photo of a gray cat and an orange dog on the grass."
prompt_placeholder  = "A dog is sitting towards the left side of a cat on a sofa in the living room. \
The color of dog is light gold and his tail is visible. The cat is white colored and has a lot of fur. \
                    A table is placed in front of the sofa. The walls of room are light orange colored."

layout_placeholder = "Caption: A realistic photo of a gray cat and an orange dog on the grass. \
Objects: [('a gray cat', [67, 243, 120, 126]), ('an orange dog', [265, 193, 190, 210])] \
Background prompt: A realistic photo of a grassy area."


layout_placeholder="Caption: Majestic Golden Retriever with light gold coat and wagging tail, delightful white \
    cat with luxurious fur, sleek table, and wall-mounted television promising relaxation and entertainment \
    Objects: [ ('a white cat', [365, 165, 123, 82]), ('a golden dog', [67, 141, 160, 203])] \
    Background prompt: Generate a garden scene"