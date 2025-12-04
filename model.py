import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block1D(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, kernel_size=3, padding=1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.activation = nn.SiLU()
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)
        
        # Add time embedding
        time_emb = self.time_mlp(t)
        # Expand time_emb to match 1D spatial dimension: (Batch, Channels, 1)
        time_emb = time_emb.unsqueeze(-1)
        h = h + time_emb
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)
        
        return h + self.shortcut(x)

class Downsample1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv1d(channels, channels, 4, 2, 1) # Stride 2 to downsample

    def forward(self, x):
        return self.op(x)

class Upsample1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.ConvTranspose1d(channels, channels, 4, 2, 1) # Stride 2 to upsample

    def forward(self, x):
        return self.op(x)

class UNet1D(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        time_emb_dim=256
    ):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.init_conv = nn.Conv1d(in_channels, base_channels, 3, padding=1)
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        channels = base_channels
        # Downsampling
        for i, mult in enumerate(channel_mults):
            out_dim = base_channels * mult
            for _ in range(2): # Two residual blocks per level
                self.downs.append(Block1D(channels, out_dim, time_emb_dim))
                channels = out_dim
            
            if i < len(channel_mults) - 1:
                self.downs.append(Downsample1D(channels))
        
        # Bottleneck
        self.mid_block1 = Block1D(channels, channels, time_emb_dim)
        self.mid_block2 = Block1D(channels, channels, time_emb_dim)
        
        # Upsampling
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_dim = base_channels * mult
            if i < len(channel_mults) - 1:
                 self.ups.append(Upsample1D(channels))
            
            for _ in range(2):
                # + out_dim because of skip connection concatenation
                # Actually, we need to track the skip connection channels carefully.
                # In this simple implementation, we assume symmetric encoder/decoder.
                # The input to the block will be cat(x, skip), so in_channels = channels + out_dim
                # Wait, let's refine the logic to match standard UNet.
                # Usually: Upsample -> Concat -> Conv
                pass

        # Re-building ModuleLists to be precise about dimensions
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        curr_channels = base_channels
        skip_connections_stack = [curr_channels] # To track skip connection sizes
        
        # Encoder
        for i, mult in enumerate(channel_mults):
            out_dim = base_channels * mult
            # Block 1
            self.downs.append(Block1D(curr_channels, out_dim, time_emb_dim))
            curr_channels = out_dim
            skip_connections_stack.append(curr_channels)
            
            # Block 2
            self.downs.append(Block1D(curr_channels, out_dim, time_emb_dim))
            curr_channels = out_dim
            skip_connections_stack.append(curr_channels)
            
            if i < len(channel_mults) - 1:
                self.downs.append(Downsample1D(curr_channels))
                skip_connections_stack.append(curr_channels)

        self.mid_block1 = Block1D(curr_channels, curr_channels, time_emb_dim)
        self.mid_block2 = Block1D(curr_channels, curr_channels, time_emb_dim)
        
        # Decoder
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_dim = base_channels * mult
            
            if i < len(channel_mults) - 1:
                self.ups.append(Upsample1D(curr_channels))
                # After upsample, we don't change channel count, but we will concat with skip
                skip_channels = skip_connections_stack.pop()
                # The skip connection for downsample is just the input to downsample
                # But wait, the skip connection stack logic needs to align with the forward pass.
                
            # We will construct the layers and handle the skip logic in forward
            # Let's just define the blocks with correct input dimensions
            
        # Let's restart the ModuleList construction to be cleaner and less error-prone
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        curr_channels = base_channels
        # We need to store what the channel size was at each step for the skip connections
        self.channel_hist = [curr_channels] 
        
        # Down
        for i, mult in enumerate(channel_mults):
            out_dim = base_channels * mult
            
            # ResBlock 1
            self.downs.append(Block1D(curr_channels, out_dim, time_emb_dim))
            curr_channels = out_dim
            self.channel_hist.append(curr_channels)
            
            # ResBlock 2
            self.downs.append(Block1D(curr_channels, out_dim, time_emb_dim))
            curr_channels = out_dim
            self.channel_hist.append(curr_channels)
            
            # Downsample
            if i < len(channel_mults) - 1:
                self.downs.append(Downsample1D(curr_channels))
                self.channel_hist.append(curr_channels)

        # Mid
        self.mid_block1 = Block1D(curr_channels, curr_channels, time_emb_dim)
        self.mid_block2 = Block1D(curr_channels, curr_channels, time_emb_dim)
        
        # Up
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_dim = base_channels * mult
            
            if i < len(channel_mults) - 1:
                self.ups.append(Upsample1D(curr_channels))
                # Pop the skip connection corresponding to the downsample
                skip_ch = self.channel_hist.pop() 
                # After upsample, we concat with the skip connection from before downsample
                # So input to next block is curr_channels + skip_ch
                
            # ResBlock 1
            skip_ch = self.channel_hist.pop()
            self.ups.append(Block1D(curr_channels + skip_ch, out_dim, time_emb_dim))
            curr_channels = out_dim
            
            # ResBlock 2
            skip_ch = self.channel_hist.pop()
            self.ups.append(Block1D(curr_channels + skip_ch, out_dim, time_emb_dim))
            curr_channels = out_dim

        self.final_conv = nn.Conv1d(curr_channels, out_channels, 1)

    def forward(self, x, t):
        # Time embedding
        t = self.time_mlp(t)
        
        x = self.init_conv(x)
        
        skips = [x]
        
        # Down
        for layer in self.downs:
            if isinstance(layer, Downsample1D):
                x = layer(x)
                skips.append(x)
            else:
                x = layer(x, t)
                skips.append(x)
                
        # Mid
        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)
        
        # Up
        # We need to consume skips in reverse order
        # The last skip is the output of the last down layer, which is 'x' currently (if we didn't have mid blocks)
        # Actually, let's trace carefully.
        # Skips structure: [Init, B1_1, B1_2, Down1, B2_1, B2_2, Down2, ...]
        # We need to pop from skips when we concat.
        
        # Let's adjust the forward loop to match the layer construction
        # The layer construction had:
        # if i < len - 1: Upsample
        # Block
        # Block
        
        # We need to pop the correct skip.
        # The last element of skips is the input to Mid.
        skips.pop() # Remove the last one because it's the input to mid, not a skip to be concatenated (or is it?)
        # Wait, standard UNet:
        # Enc: x -> [Block] -> s1 -> [Down] -> x2
        # Dec: x2 -> [Up] -> u1; cat(u1, s1) -> [Block]
        
        # My skips list contains outputs of EVERY layer in downs.
        # Let's iterate through self.ups
        
        for layer in self.ups:
            if isinstance(layer, Upsample1D):
                x = layer(x)
                # The skip to concat is the one BEFORE the corresponding Downsample
                # In my skips list, I appended after every layer.
                # So if I just pop(), I get the output of the layer that fed into the current level?
                
                # Let's look at the construction again.
                # Down loop:
                #   Block -> append to skips
                #   Block -> append to skips
                #   Downsample -> append to skips
                
                # Up loop:
                #   Upsample (corresponds to Downsample) -> Pop skip (which is the input to Downsample, i.e. output of last Block)
                #   Block (corresponds to Block 2) -> Pop skip (output of Block 1)
                #   Block (corresponds to Block 1) -> Pop skip (output of previous level or init)
                
                # So yes, just popping should work if the order is exact mirror.
                # But wait, Downsample output was appended. We need to pop that first?
                # No, the output of Downsample is the input to the next level.
                # The skip connection for the Upsample layer is usually the feature map BEFORE downsampling.
                # So we need to pop the skip that corresponds to "Before Downsample".
                # The "After Downsample" skip is effectively the input to the lower level, which we processed.
                
                skip = skips.pop() # This is the output of Downsample?
                # If the last thing in downs was Downsample, then yes.
                # But we want the one BEFORE Downsample.
                # So we might need to discard the one that represents "After Downsample" if we stored it.
                pass
            else:
                # It's a Block
                skip = skips.pop()
                x = torch.cat((x, skip), dim=1)
                x = layer(x, t)
                
        return self.final_conv(x)

# Re-writing the class to be cleaner and simpler with explicit skip management
class UNet1D_Clean(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        time_emb_dim=256
    ):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.init_conv = nn.Conv1d(in_channels, base_channels, 3, padding=1)
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        curr_channels = base_channels
        
        # Downsampling path
        for i, mult in enumerate(channel_mults):
            out_dim = base_channels * mult
            # Two ResBlocks per level
            self.downs.append(nn.ModuleList([
                Block1D(curr_channels, out_dim, time_emb_dim),
                Block1D(out_dim, out_dim, time_emb_dim)
            ]))
            curr_channels = out_dim
            
            # Downsample (except last level)
            if i < len(channel_mults) - 1:
                self.downs.append(Downsample1D(curr_channels))
        
        # Bottleneck
        self.mid_block1 = Block1D(curr_channels, curr_channels, time_emb_dim)
        self.mid_block2 = Block1D(curr_channels, curr_channels, time_emb_dim)
        
        # Upsampling path
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_dim = base_channels * mult
            
            # Upsample (except first level of decoder which corresponds to last level of encoder)
            if i < len(channel_mults) - 1:
                self.ups.append(Upsample1D(curr_channels))
                # After upsample, we concat with skip from the corresponding down level
                # The skip size is 'out_dim' (the size before downsampling)
                cat_dim = curr_channels + out_dim # curr is from lower level, out_dim is from skip
                
                # Wait, if we upsample, the channel count stays same.
                # So it is curr_channels + skip_channels.
                # In the symmetric encoder, the skip channel size is 'out_dim'.
                # So input to next block is curr_channels + out_dim.
            else:
                # For the deepest level, we don't upsample, we just start processing
                # But wait, the deepest level in 'downs' didn't have a downsample.
                # So we are just moving back up.
                # Let's stick to the structure:
                # Down: [Block, Block, Downsample]
                # Up: [Upsample, Block, Block]
                pass

            # We need to be very precise.
            # Let's use a simpler structure: explicit layers in a list is hard to align without running it.
            # I will use the "Clean" structure but define the layers in the loop carefully.
            pass

class UNet1D_Final(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        time_emb_dim=256
    ):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Magnitude Embedding
        self.mag_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.init_conv = nn.Conv1d(in_channels, base_channels, 3, padding=1)
        
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        
        # Down
        curr_channels = base_channels
        skip_dims = [curr_channels]
        
        for i, mult in enumerate(channel_mults):
            out_dim = base_channels * mult
            
            # Block 1
            block1 = Block1D(curr_channels, out_dim, time_emb_dim)
            # Block 2
            block2 = Block1D(out_dim, out_dim, time_emb_dim)
            
            # Downsample or Identity
            if i < len(channel_mults) - 1:
                down = Downsample1D(out_dim)
                self.down_blocks.append(nn.ModuleList([block1, block2, down]))
                skip_dims.append(out_dim) # For block 2 output
            else:
                self.down_blocks.append(nn.ModuleList([block1, block2, nn.Identity()]))
                skip_dims.append(out_dim)
            
            curr_channels = out_dim

        # Mid
        self.mid_block1 = Block1D(curr_channels, curr_channels, time_emb_dim)
        self.mid_block2 = Block1D(curr_channels, curr_channels, time_emb_dim)
        
        # Up
        # skip_dims has the channel sizes of the skip connections.
        # We need to process in reverse.
        
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_dim = base_channels * mult
            
            # We need to know what we are concatenating with.
            # The skip connection comes from the corresponding Down block.
            # The Down block outputted 'out_dim' channels (before downsampling).
            skip_ch = skip_dims.pop()
            
            if i < len(channel_mults) - 1:
                upsample = Upsample1D(curr_channels)
            else:
                upsample = nn.Identity()
                
            # After upsample (if any), we concat with skip_ch.
            # So input to block is curr_channels + skip_ch
            
            block1 = Block1D(curr_channels + skip_ch, out_dim, time_emb_dim)
            block2 = Block1D(out_dim, out_dim, time_emb_dim)
            
            self.up_blocks.append(nn.ModuleList([upsample, block1, block2]))
            curr_channels = out_dim
            
        self.final_conv = nn.Conv1d(curr_channels, out_channels, 1)

    def forward(self, x, t, y=None):
        t = self.time_mlp(t)
        
        if y is not None:
            # y is (Batch,) or (Batch, 1)
            if len(y.shape) == 1:
                y = y.unsqueeze(-1)
            mag_emb = self.mag_mlp(y)
            t = t + mag_emb
            
        x = self.init_conv(x)
        
        skips = []
        
        # Down
        for block1, block2, down in self.down_blocks:
            x = block1(x, t)
            x = block2(x, t)
            skips.append(x) # Save output before downsampling
            x = down(x)
            
        # Mid
        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)
        
        # Up
        for upsample, block1, block2 in self.up_blocks:
            x = upsample(x)
            skip = skips.pop()
            
            # Handle potential size mismatch due to odd padding/striding
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode='linear', align_corners=False)
                
            x = torch.cat((x, skip), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            
        return self.final_conv(x)
