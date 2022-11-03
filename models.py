import torch
from torch import nn

class BaseNet(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(in_size, 1024), nn.ReLU(), nn.Dropout(0.2), nn.BatchNorm1d(1024),
            nn.Linear(1024, 128), nn.ReLU(), nn.Dropout(0.2), nn.BatchNorm1d(128),
            nn.Linear(128, out_size), nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.classifier(x)
        return x

class EncoderNet(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        latent_dim = 32
        h_dim = 256

        self.mu_encoder = nn.Sequential(
            nn.Linear(in_size, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

        self.net = nn.Sequential(
            nn.Linear(latent_dim, h_dim), nn.BatchNorm1d(h_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h_dim, out_size), nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.mu_encoder(x)
        x = self.net(x)
        return x

class MultiEncoderNet(nn.Module):
    def __init__(self, h_dim, out_size):
        super().__init__()

        # self.sbs_rd_encoder = nn.Sequential(
        #     nn.Linear(3113, 1024), nn.ReLU(), nn.BatchNorm1d(1024)
        # )

        # self.indel_rd_encoder = nn.Sequential(
        #     nn.Linear(3113, 1024), nn.ReLU(), nn.BatchNorm1d(1024)
        # )

        # self.cnv_rd_encoder = nn.Sequential(
        #     nn.Linear(3113, 1024), nn.ReLU(), nn.BatchNorm1d(1024)
        # )

        self.sbs_encoder = nn.Sequential(
            nn.Linear(3209, 256), nn.ReLU(), nn.BatchNorm1d(256)
        )

        self.indel_encoder = nn.Sequential(
            nn.Linear(3209, 256), nn.ReLU(), nn.BatchNorm1d(256)
        )

        self.cnv_encoder = nn.Sequential(
            nn.Linear(3209, 256), nn.ReLU(), nn.BatchNorm1d(256)
        )

        self.h_dim = h_dim

        self.mu_encoder = nn.Sequential(
            nn.Linear(768, self.h_dim)
        )
    
        self.clf = nn.Sequential(
            # nn.Linear(self.h_dim, 128), nn.ReLU(), nn.BatchNorm1d(128),
            # nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.2), nn.BatchNorm1d(32),
            nn.Linear(self.h_dim, out_size), nn.Softmax(dim=1)
        )

    def encode(self, inputs):
        # sbs_sig   = inputs[:, :96]
        # sbs_rd    = inputs[:, 96 : 96+3113]
        # indel_sig = inputs[:, 96+3113 : 96+3113+96]
        # indel_rd  = inputs[:, 96+3113+96 : 96+3113+96+3113]
        # cnv_sig   = inputs[:, 96+3113+96+3113 : 96+3113+96+3113+96]
        # cnv_rd    = inputs[:, 96+3113+96+3113+96:]


        sbs   = inputs[:, :3209]
        indel = inputs[:, 3209:3209*2]
        cnv   = inputs[:, 3209*2:]

        # print(len(sbs_rd[0]), len(indel_rd[0]), len(cnv_rd[0]))

        # print(len(cnv_sig[0]))

        # sbs_rd_enc   = self.sbs_rd_encoder(sbs_rd)
        # indel_rd_enc = self.indel_rd_encoder(indel_rd)
        # cnv_rd_enc   = self.cnv_rd_encoder(cnv_rd)

        # sbs_fus   = torch.concat([sbs_sig,   sbs_rd_enc],   dim=1)
        # indel_fus = torch.concat([indel_sig, indel_rd_enc], dim=1)
        # cnv_fus   = torch.concat([cnv_sig,   cnv_rd_enc],   dim=1)

        sbs_enc   = self.sbs_encoder(sbs)
        indel_enc = self.indel_encoder(indel)
        cnv_enc   = self.cnv_encoder(cnv)

        total_enc = torch.concat([sbs_enc, indel_enc, cnv_enc], dim=1)

        # mu, logvar = torch.split(self.v_encoder(total_enc), self.h_dim, dim=1)
        mu     = self.mu_encoder(total_enc)
        # logvar = self.logvar_encoder(total_enc)

        return mu
        
    def forward(self, inputs):
        z = self.encode(inputs)
        probs = self.clf(z)
        return probs

class MultiEncoderNetPartial(nn.Module):
    def __init__(self, out_size):
        super().__init__()

        self.sbs_rd_encoder = nn.Sequential(
            nn.Linear(3113, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 64)
        )

        self.indel_rd_encoder = nn.Sequential(
            nn.Linear(6226, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 64)
        )

        self.sbs_encoder = nn.Sequential(
            nn.Linear(160, 64), nn.BatchNorm1d(64), nn.ReLU()
        )

        self.indel_encoder = nn.Sequential(
            nn.Linear(160, 64), nn.BatchNorm1d(64), nn.ReLU()
        )

        self.mu_encoder = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(32, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, out_size), nn.Softmax(dim=1)
        )

    def encode(self, inputs):
        sbs_sig   = inputs[0]
        sbs_rd    = inputs[1]
        indel_sig = inputs[2]
        indel_rd  = inputs[3]

        # print(len(sbs_rd[0]), len(indel_rd[0]), len(cnv_rd[0]))

        # print(len(cnv_sig[0]))

        sbs_rd_enc   = self.sbs_rd_encoder(sbs_rd)
        indel_rd_enc = self.indel_rd_encoder(indel_rd)

        sbs_fus   = torch.concat([sbs_sig,   sbs_rd_enc],   dim=1)
        indel_fus = torch.concat([indel_sig, indel_rd_enc], dim=1)

        sbs_enc   = self.sbs_encoder(sbs_fus)
        indel_enc = self.indel_encoder(indel_fus)

        total_enc = torch.concat([sbs_enc, indel_enc], dim=1)

        mu = self.mu_encoder(total_enc)

        return mu
        
    def forward(self, inputs):
        mu = self.encode(inputs)
        probs = self.classifier(mu)
        return probs

class LinearVAE(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        h = 32

        self.mu_encoder = nn.Sequential(
            nn.Linear(in_size, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, h)
        )

        self.logvar_encoder = nn.Sequential(
            nn.Linear(in_size, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, h)
        )

        self.z_decoder = nn.Sequential(
            nn.Linear(h, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, in_size), nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.rand_like(std)
        sample = mu + (eps * std)
        return sample

    def forward(self, x):
        mu     = self.mu_encoder(x)
        logvar = self.logvar_encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.z_decoder(z)
        return recon, mu, logvar

class MultiVAE(nn.Module):
    def __init__(self, h_dim):
        super().__init__()

        # self.sbs_rd_encoder = nn.Sequential(
        #     nn.Linear(3113, 1024), nn.ReLU(), nn.BatchNorm1d(1024)
        # )

        # self.indel_rd_encoder = nn.Sequential(
        #     nn.Linear(3113, 1024), nn.ReLU(), nn.BatchNorm1d(1024)
        # )

        # self.cnv_rd_encoder = nn.Sequential(
        #     nn.Linear(3113, 1024), nn.ReLU(), nn.BatchNorm1d(1024)
        # )

        self.sbs_encoder = nn.Sequential(
            nn.Linear(3209, 256), nn.ReLU(), nn.BatchNorm1d(256)
        )

        self.indel_encoder = nn.Sequential(
            nn.Linear(3209, 256), nn.ReLU(), nn.BatchNorm1d(256)
        )

        self.cnv_encoder = nn.Sequential(
            nn.Linear(3209, 256), nn.ReLU(), nn.BatchNorm1d(256)
        )

        self.h_dim = h_dim

        self.mu_encoder = nn.Sequential(
            nn.Linear(768, self.h_dim)
        )

        #############################################################

        self.logvar_encoder = nn.Sequential(
            nn.Linear(768, self.h_dim)
        )

        self.z_decoder = nn.Sequential(
            nn.Linear(self.h_dim, 768)
        )

        self.sbs_decoder = nn.Sequential(
            nn.Linear(256, 3209), nn.Sigmoid()
        )

        self.indel_decoder = nn.Sequential(
            nn.Linear(256, 3209), nn.Sigmoid()
        )

        self.cnv_decoder = nn.Sequential(
            nn.Linear(256, 3209), nn.Sigmoid()
        )

        # self.sbs_rd_decoder = nn.Sequential(
        #     nn.Linear(1024, 3113), nn.Sigmoid()
        # )

        # self.indel_rd_decoder = nn.Sequential(
        #     nn.Linear(1024, 3113), nn.Sigmoid()
        # )

        # self.cnv_rd_decoder = nn.Sequential(
        #     nn.Linear(1024, 3113), nn.Sigmoid()
        # )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.rand_like(std)
        sample = mu + (eps * std)
        return sample

    def encode(self, inputs):
        # sbs_sig   = inputs[:, :96]
        # sbs_rd    = inputs[:, 96 : 96+3113]
        # indel_sig = inputs[:, 96+3113 : 96+3113+96]
        # indel_rd  = inputs[:, 96+3113+96 : 96+3113+96+3113]
        # cnv_sig   = inputs[:, 96+3113+96+3113 : 96+3113+96+3113+96]
        # cnv_rd    = inputs[:, 96+3113+96+3113+96:]

        sbs   = inputs[:, :3209]
        indel = inputs[:, 3209:3209*2]
        cnv   = inputs[:, 3209*2:]

        # print(len(sbs_rd[0]), len(indel_rd[0]), len(cnv_rd[0]))

        # print(len(cnv_sig[0]))

        # sbs_rd_enc   = self.sbs_rd_encoder(sbs_rd)
        # indel_rd_enc = self.indel_rd_encoder(indel_rd)
        # cnv_rd_enc   = self.cnv_rd_encoder(cnv_rd)

        # sbs_fus   = torch.concat([sbs_sig,   sbs_rd_enc],   dim=1)
        # indel_fus = torch.concat([indel_sig, indel_rd_enc], dim=1)
        # cnv_fus   = torch.concat([cnv_sig,   cnv_rd_enc],   dim=1)

        sbs_enc   = self.sbs_encoder(sbs)
        indel_enc = self.indel_encoder(indel)
        cnv_enc   = self.cnv_encoder(cnv)

        total_enc = torch.concat([sbs_enc, indel_enc, cnv_enc], dim=1)

        # mu, logvar = torch.split(self.v_encoder(total_enc), self.h_dim, dim=1)
        mu     = self.mu_encoder(total_enc)
        logvar = self.logvar_encoder(total_enc)

        return mu, logvar

    def forward(self, inputs):
        # sbs_sig   = inputs[:, :96]
        # # sbs_rd    = inputs[:, 96 : 96+3113]
        # indel_sig = inputs[:, 96+3113 : 96+3113+96]
        # # # indel_rd  = inputs[:, 96+3113+96 : 96+3113+96+3113]
        # cnv_sig   = inputs[:, 96+3113+96+3113 : 96+3113+96+3113+96]
        # cnv_rd    = inputs[:, 96+3113+96+3113+96:]
        # print(len(inputs), len(inputs[0]), len(inputs[0][0]))

        mu, logvar = self.encode(inputs)
        # logvar = self.logvar_encoder(total_enc)

        z = self.reparameterize(mu, logvar)

        total_dec = self.z_decoder(z)

        chunks = torch.chunk(total_dec, chunks=3, dim=1)

        sbs_fus_dec   = self.sbs_decoder(chunks[0])
        indel_fus_dec = self.indel_decoder(chunks[1])
        cnv_fus_dec   = self.cnv_decoder(chunks[2])

        # sbs_dec   = self.sbs_final_decoder(sbs_fus)
        # indel_dec = self.indel_final_decoder(indel_fus)
        # cnv_dec   = self.cnv_final_decoder(cnv_fus)

        # sbs_sig_dec   = sbs_fus_dec[:, :len(sbs_sig[0])]
        # indel_sig_dec = indel_fus_dec[:, :len(indel_sig[0])]
        # cnv_sig_dec   = cnv_fus_dec[:, :len(cnv_sig[0])]

        # sbs_rd_dec    = self.sbs_rd_decoder(sbs_fus_dec[:, len(sbs_sig[0]):])
        # indel_rd_dec  = self.indel_rd_decoder(indel_fus_dec[:, len(indel_sig[0]):])
        # cnv_rd_dec    = self.cnv_rd_decoder(cnv_fus_dec[:, len(cnv_sig[0]):])

        # fus_inputs  = total_enc
        # fus_outputs = total_dec
        # final_outputs = torch.concat([sbs_sig_dec, sbs_rd_dec, indel_sig_dec, indel_rd_dec, cnv_sig_dec, cnv_rd_dec], dim=1)
        final_outputs = torch.concat([sbs_fus_dec, indel_fus_dec, cnv_fus_dec], dim=1)
        
        return final_outputs, mu, logvar

class SupervisedMultiVAE(nn.Module):
    def __init__(self, out_size):
        super().__init__()

        self.sbs_rd_encoder = nn.Sequential(
            # nn.Linear(3113, 256), nn.BatchNorm1d(256), nn.ReLU(),
            # nn.Linear(256, 64)
            nn.Linear(3113, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.ReLU()
        )

        self.indel_rd_encoder = nn.Sequential(
            # nn.Linear(6226, 256), nn.BatchNorm1d(256), nn.ReLU(),
            # nn.Linear(256, 64)
            nn.Linear(3113, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.ReLU()
        )

        self.cnv_rd_encoder = nn.Sequential(
            # nn.Linear(3113, 256), nn.BatchNorm1d(256), nn.ReLU(),
            # nn.Linear(256, 64)
            nn.Linear(3113, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.ReLU()
        )

        self.sbs_encoder = nn.Sequential(
            nn.Linear(352, 128), nn.BatchNorm1d(128), nn.Sigmoid()
            # nn.Linear(160, 64), nn.BatchNorm1d(64), nn.ReLU()
        )

        self.indel_encoder = nn.Sequential(
            # 160
            nn.Linear(352, 128), nn.BatchNorm1d(128), nn.Sigmoid()
        )

        self.cnv_encoder = nn.Sequential(
            nn.Linear(352, 128), nn.BatchNorm1d(128), nn.Sigmoid()
        )

        self.h_dim = 128

        self.v_encoder = nn.Sequential(
            # nn.Linear(192, 96), nn.BatchNorm1d(96), nn.ReLU(),
            # nn.Linear(96, 32)
            nn.Linear(384, self.h_dim*2)
        )

        self.clf = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, out_size), nn.Softmax(dim=1)
        )

        # self.logvar_encoder = nn.Sequential(
        #     # nn.Linear(192, 96), nn.BatchNorm1d(96), nn.ReLU(),
        #     # nn.Linear(96, 32)
        #     nn.Linear(192, 128)
        # )

        self.z_decoder = nn.Sequential(
            # nn.Linear(32, 96), nn.BatchNorm1d(96), nn.ReLU(),
            # nn.Linear(96, 192)
            nn.Linear(self.h_dim, 384), nn.BatchNorm1d(384), nn.Sigmoid()
        )

        self.sbs_decoder = nn.Sequential(
            # nn.Linear(64, 160), nn.BatchNorm1d(160), nn.Sigmoid()
            nn.Linear(128, 352), nn.Sigmoid()
        )

        self.indel_decoder = nn.Sequential(
            # nn.Linear(64, 160), nn.BatchNorm1d(160), nn.Sigmoid()
            nn.Linear(128, 352), nn.Sigmoid()
        )

        self.cnv_decoder = nn.Sequential(
            # nn.Linear(64, 160), nn.BatchNorm1d(160), nn.Sigmoid()
            nn.Linear(128, 352), nn.Sigmoid()
        )

        self.sbs_rd_decoder = nn.Sequential(
            # nn.Linear(64, 256), nn.BatchNorm1d(256), nn.ReLU(),
            # nn.Linear(256, 3113), nn.Sigmoid()
            nn.Linear(256, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024, 3113), nn.Sigmoid()
        )

        self.indel_rd_decoder = nn.Sequential(
            # nn.Linear(64, 256), nn.BatchNorm1d(256), nn.ReLU(),
            # nn.Linear(256, 6226), nn.Sigmoid()
            nn.Linear(256, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024, 3113), nn.Sigmoid()
        )

        self.cnv_rd_decoder = nn.Sequential(
            # nn.Linear(64, 256), nn.BatchNorm1d(256), nn.ReLU(),
            # nn.Linear(256, 3113), nn.Sigmoid()
            nn.Linear(256, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024, 3113), nn.Sigmoid()
        )

        # self.sbs_final_decoder = nn.Sequential(
        #     nn.Linear(160, 256), nn.BatchNorm1d(256), nn.ReLU(),
        #     nn.Linear(256, 96+3113), nn.Sigmoid()
        # )

        # self.indel_final_decoder = nn.Sequential(
        #     nn.Linear(160, 256), nn.BatchNorm1d(256), nn.ReLU(),
        #     nn.Linear(256, 96+6226), nn.Sigmoid()
        # )

        # self.cnv_final_decoder = nn.Sequential(
        #     nn.Linear(160, 256), nn.BatchNorm1d(256), nn.ReLU(),
        #     nn.Linear(256, 11+3113), nn.Sigmoid()
        # )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.rand_like(std)
        sample = mu + (eps * std)
        return sample

    # def inference(self, inputs):
    #     return self.clf(inputs)

    def encode(self, inputs):
        sbs_sig   = inputs[:, :96]
        sbs_rd    = inputs[:, 96 : 96+3113]
        indel_sig = inputs[:, 96+3113 : 96+3113+96]
        indel_rd  = inputs[:, 96+3113+96 : 96+3113+96+3113]
        cnv_sig   = inputs[:, 96+3113+96+3113 : 96+3113+96+3113+96]
        cnv_rd    = inputs[:, 96+3113+96+3113+96:]

        # print(len(sbs_rd[0]), len(indel_rd[0]), len(cnv_rd[0]))

        # print(len(cnv_sig[0]))

        sbs_rd_enc   = self.sbs_rd_encoder(sbs_rd)
        indel_rd_enc = self.indel_rd_encoder(indel_rd)
        cnv_rd_enc   = self.cnv_rd_encoder(cnv_rd)

        sbs_fus   = torch.concat([sbs_sig,   sbs_rd_enc],   dim=1)
        indel_fus = torch.concat([indel_sig, indel_rd_enc], dim=1)
        cnv_fus   = torch.concat([cnv_sig,   cnv_rd_enc],   dim=1)

        sbs_enc   = self.sbs_encoder(sbs_fus)
        indel_enc = self.indel_encoder(indel_fus)
        cnv_enc   = self.cnv_encoder(cnv_fus)

        total_enc = torch.concat([sbs_enc, indel_enc, cnv_enc], dim=1)

        mu, logvar = torch.split(self.v_encoder(total_enc), self.h_dim, dim=1)

        return mu, logvar

    def forward(self, inputs):
        # print(len(inputs), len(inputs[0]), len(inputs[0][0]))

        # sbs_sig   = inputs[0]
        # sbs_rd    = inputs[1]
        # indel_sig = inputs[2]
        # indel_rd  = inputs[3]
        # cnv_sig   = inputs[4]
        # cnv_rd    = inputs[5]

        
        # logvar = self.logvar_encoder(total_enc)

        mu, logvar = self.encode(inputs)

        # Classifier
        probs = self.clf(mu)

        z = self.reparameterize(mu, logvar)

        total_dec = self.z_decoder(z)

        chunks = torch.chunk(total_dec, chunks=3, dim=1)

        sbs_fus_dec   = self.sbs_decoder(chunks[0])
        indel_fus_dec = self.indel_decoder(chunks[1])
        cnv_fus_dec   = self.cnv_decoder(chunks[2])

        # sbs_dec   = self.sbs_final_decoder(sbs_fus)
        # indel_dec = self.indel_final_decoder(indel_fus)
        # cnv_dec   = self.cnv_final_decoder(cnv_fus)

        sbs_sig_dec   = sbs_fus_dec[:, :len(sbs_sig[0])]
        indel_sig_dec = indel_fus_dec[:, :len(indel_sig[0])]
        cnv_sig_dec   = cnv_fus_dec[:, :len(cnv_sig[0])]

        sbs_rd_dec    = self.sbs_rd_decoder(sbs_fus_dec[:, len(sbs_sig[0]):])
        indel_rd_dec  = self.indel_rd_decoder(indel_fus_dec[:, len(indel_sig[0]):])
        cnv_rd_dec    = self.cnv_rd_decoder(cnv_fus_dec[:, len(cnv_sig[0]):])

        fus_inputs  = total_enc
        fus_outputs = total_dec
        final_outputs = torch.concat([sbs_sig_dec, sbs_rd_dec, indel_sig_dec, indel_rd_dec, cnv_sig_dec, cnv_rd_dec], dim=1)
        
        return probs, fus_inputs, fus_outputs, final_outputs, mu, logvar

class MultiVAE_Partial(nn.Module):
    def __init__(self):
        super().__init__()

        self.sbs_rd_encoder = nn.Sequential(
            nn.Linear(3113, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 64)
        )

        self.indel_rd_encoder = nn.Sequential(
            nn.Linear(6226, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 64)
        )

        # self.cnv_rd_encoder = nn.Sequential(
        #     nn.Linear(3113, 256), nn.BatchNorm1d(256), nn.ReLU(),
        #     nn.Linear(256, 64)
        # )

        self.sbs_encoder = nn.Sequential(
            nn.Linear(160, 64), nn.BatchNorm1d(64), nn.ReLU()
        )

        self.indel_encoder = nn.Sequential(
            nn.Linear(160, 64), nn.BatchNorm1d(64), nn.ReLU()
        )

        # self.cnv_encoder = nn.Sequential(
        #     nn.Linear(75, 64), nn.BatchNorm1d(64), nn.ReLU()
        # )

        self.mu_encoder = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 32)
        )

        self.logvar_encoder = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 32)
        )

        self.z_decoder = nn.Sequential(
            nn.Linear(32, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 128)
        )

        self.sbs_decoder = nn.Sequential(
            nn.Linear(64, 160), nn.Sigmoid()
        )

        self.indel_decoder = nn.Sequential(
            nn.Linear(64, 160), nn.Sigmoid()
        )

        # self.cnv_decoder = nn.Sequential(
        #     nn.Linear(64, 160), nn.BatchNorm1d(160), nn.ReLU()
        # )

        self.sbs_rd_decoder = nn.Sequential(
            nn.Linear(64, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 3113), nn.Sigmoid()
        )

        self.indel_rd_decoder = nn.Sequential(
            nn.Linear(64, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 6226), nn.Sigmoid()
        )

        # self.cnv_final_decoder = nn.Sequential(
        #     nn.Linear(160, 256), nn.BatchNorm1d(256), nn.ReLU(),
        #     nn.Linear(256, 11+3113), nn.Sigmoid()
        # )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.rand_like(std)
        sample = mu + (eps * std)
        return sample

    def forward(self, inputs):
        # print(len(inputs), len(inputs[0]), len(inputs[0][0]))

        # sbs_sig   = inputs[0]
        # sbs_rd    = inputs[1]
        # indel_sig = inputs[2]
        # indel_rd  = inputs[3]
        # cnv_sig   = inputs[4]
        # cnv_rd    = inputs[5]

        # print(len(sbs_rd[0]), len(indel_rd[0]), len(cnv_rd[0]))

        # print(len(cnv_sig[0]))

        sbs_rd_enc   = self.sbs_rd_encoder(sbs_rd)
        indel_rd_enc = self.indel_rd_encoder(indel_rd)
        # cnv_rd_enc   = self.cnv_rd_encoder(cnv_rd)

        sbs_fus   = torch.concat([sbs_sig,   sbs_rd_enc],   dim=1)
        indel_fus = torch.concat([indel_sig, indel_rd_enc], dim=1)
        # cnv_fus   = torch.concat([cnv_sig,   cnv_rd_enc],   dim=1)

        sbs_enc   = self.sbs_encoder(sbs_fus)
        indel_enc = self.indel_encoder(indel_fus)
        # cnv_enc   = self.cnv_encoder(cnv_fus)

        total_enc = torch.concat([sbs_enc, indel_enc], dim=1)

        mu     = self.mu_encoder(total_enc)
        logvar = self.logvar_encoder(total_enc)

        z = self.reparameterize(mu, logvar)

        total_dec = self.z_decoder(z)

        chunks = torch.chunk(total_dec, chunks=2, dim=1)

        sbs_fus_dec   = self.sbs_decoder(chunks[0])
        indel_fus_dec = self.indel_decoder(chunks[1])
        # cnv_fus   = self.cnv_decoder(chunks[2])

        sbs_sig_dec   = sbs_fus_dec[:, :len(sbs_sig[0])]
        indel_sig_dec = indel_fus_dec[:, :len(indel_sig[0])]
        sbs_rd_dec    = self.sbs_rd_decoder(sbs_fus_dec[:, len(sbs_sig[0]):])
        indel_rd_dec  = self.indel_rd_decoder(indel_fus_dec[:, len(indel_sig[0]):])
        # cnv_dec   = self.cnv_final_decoder(cnv_fus)

        # for x in [sbs_sig_dec, indel_sig_dec, sbs_rd_dec, indel_rd_dec]:
            # print(x.size())

        outputs = torch.concat([sbs_sig_dec, indel_sig_dec, sbs_rd_dec, indel_rd_dec], dim=1)
        
        return outputs, mu, logvar