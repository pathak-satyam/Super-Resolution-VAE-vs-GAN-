"""
Create comparison visualization for SRVAE vs SRGAN
"""

import matplotlib.pyplot as plt
import numpy as np

methods = ['SRVAE', 'SRGAN']
psnr_values = [26.19, 30.42]
ssim_values = [0.7686, 0.8973]
mse_values = [0.002419, 0.000921]

# Colors
colors = ['#5DADE2', '#FF69B4']  # Cyan for VAE, Pink for GAN

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor('#2C2C2C')  # Dark background

# Plot 1: PSNR
ax1 = axes[0]
bars1 = ax1.bar(methods, psnr_values, color=colors, edgecolor='white', linewidth=2)
ax1.set_ylabel('PSNR (dB)', fontsize=13, color='white', fontweight='bold')
ax1.set_title('Peak Signal-to-Noise Ratio\n(Higher is Better)',
              fontsize=14, color='white', fontweight='bold', pad=15)
ax1.set_ylim(0, max(psnr_values) * 1.15)
ax1.tick_params(colors='white', labelsize=12)
ax1.set_facecolor('#1E1E1E')
ax1.spines['bottom'].set_color('white')
ax1.spines['left'].set_color('white')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(True, alpha=0.2, color='white', linestyle='-', linewidth=0.5)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars1, psnr_values)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{val:.2f}', ha='center', va='bottom',
             color='white', fontsize=12, fontweight='bold')

# Plot 2: SSIM
ax2 = axes[1]
bars2 = ax2.bar(methods, ssim_values, color=colors, edgecolor='white', linewidth=2)
ax2.set_ylabel('SSIM', fontsize=13, color='white', fontweight='bold')
ax2.set_title('Structural Similarity Index\n(Higher is Better, max=1)',
              fontsize=14, color='white', fontweight='bold', pad=15)
ax2.set_ylim(0, 1.05)
ax2.tick_params(colors='white', labelsize=12)
ax2.set_facecolor('#1E1E1E')
ax2.spines['bottom'].set_color('white')
ax2.spines['left'].set_color('white')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(True, alpha=0.2, color='white', linestyle='-', linewidth=0.5)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars2, ssim_values)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{val:.4f}', ha='center', va='bottom',
             color='white', fontsize=12, fontweight='bold')

# Plot 3: MSE
ax3 = axes[2]
bars3 = ax3.bar(methods, mse_values, color=colors, edgecolor='white', linewidth=2)
ax3.set_ylabel('MSE', fontsize=13, color='white', fontweight='bold')
ax3.set_title('Mean Squared Error\n(Lower is Better)',
              fontsize=14, color='white', fontweight='bold', pad=15)
ax3.set_ylim(0, max(mse_values) * 1.15)
ax3.tick_params(colors='white', labelsize=12)
ax3.set_facecolor('#1E1E1E')
ax3.spines['bottom'].set_color('white')
ax3.spines['left'].set_color('white')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.grid(True, alpha=0.2, color='white', linestyle='-', linewidth=0.5)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars3, mse_values)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
             f'{val:.6f}', ha='center', va='bottom',
             color='white', fontsize=12, fontweight='bold')

# Add improvement percentage on SRGAN bars
improvement_psnr = ((psnr_values[1] - psnr_values[0]) / psnr_values[0]) * 100
improvement_ssim = ((ssim_values[1] - ssim_values[0]) / ssim_values[0]) * 100
improvement_mse = ((mse_values[0] - mse_values[1]) / mse_values[0]) * 100

ax1.text(bars1[1].get_x() + bars1[1].get_width()/2., psnr_values[1] - 2,
         f'+{improvement_psnr:.1f}%', ha='center', va='top',
         color='white', fontsize=10, fontweight='bold', style='italic')

ax2.text(bars2[1].get_x() + bars2[1].get_width()/2., ssim_values[1] - 0.05,
         f'+{improvement_ssim:.1f}%', ha='center', va='top',
         color='white', fontsize=10, fontweight='bold', style='italic')

ax3.text(bars3[1].get_x() + bars3[1].get_width()/2., mse_values[1] + 0.0003,
         f'-{improvement_mse:.1f}%', ha='center', va='bottom',
         color='white', fontsize=10, fontweight='bold', style='italic')

plt.tight_layout()
plt.savefig('srvae_vs_srgan_comparison.png', dpi=300, facecolor='#2C2C2C',
            bbox_inches='tight', edgecolor='none')
plt.show()

print("✓ Visualization saved as 'srvae_vs_srgan_comparison.png'")
print("\nSummary:")
print(f"  SRGAN outperforms SRVAE by:")
print(f"  • PSNR: +{improvement_psnr:.1f}% (+{psnr_values[1] - psnr_values[0]:.2f} dB)")
print(f"  • SSIM: +{improvement_ssim:.1f}% (+{ssim_values[1] - ssim_values[0]:.4f})")
print(f"  • MSE:  -{improvement_mse:.1f}% (lower is better)")