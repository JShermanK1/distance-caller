#![feature(const_cstr_methods)]

use noodles::{
    bcf::{self, header::StringMaps},
    vcf::Header
};
use polars::prelude::*;
use std::{
    error::Error,
    fs::File,
    iter::{repeat},
    path::PathBuf,
};
use itertools::Itertools;
use noodles::csi;
use rayon::prelude::*;
use clap::{
    Command, 
    arg, 
    value_parser
};
use sync_file::SyncFile;
use tikv_jemallocator::Jemalloc;


#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;


fn cli() -> Command {
    Command::new("infochallenge")
            .args(&[
                arg!(
                    -i --in <FILE> "Path to bcf file with index"
                ).required(true)
                 .value_parser(value_parser!(PathBuf)),
            ])
}


fn main() -> Result<(), Box<dyn Error>> {

    



    let matches = cli().get_matches();
    let in_path = matches.get_one::<PathBuf>("in").expect("error parsing input path");
    let mut index_path = in_path.clone();
    index_path.set_extension("bcf.csi");


    let mut index_r = File::open(index_path).map(csi::Reader::new)
                            .expect("Could not find index");
    let index = index_r.read_index()?;

    let f = SyncFile::open(in_path)?;

    let num_samples;
    let stringmaps: StringMaps;
    let header: Header;

    {
        let mut bcf_r = bcf::Reader::new(f.clone());

        bcf_r.read_file_format()?;
        let r_header = bcf_r.read_header()?;
        header = r_header.parse()?;
        stringmaps = r_header.parse()?;

        let mut records = bcf_r.records().peekable();

        let first_record = records.peek().unwrap().clone();
        num_samples = first_record.as_ref().unwrap().genotypes().len();
    }

    let sample_names = header.sample_names()
                             .iter()
                             .map(|string| 
                                string.split('/')
                                      .last().unwrap()
                                      .split(".").next().unwrap()
                             ).collect::<Vec<&str>>();

    let names_top = "\t".to_string() + sample_names
                                            .iter()
                                            .join("\t")
                                            .as_str();

    println!("{names_top}");

    let sizes = index.reference_sequences()
                     .par_iter()
                     .map(|refseq| {
                        let size;
                        if let Some(meta) = refseq.metadata() {
                            size = Some(meta.mapped_record_count() as usize);
                        } else {
                            size = None
                        }

                        size

                     });

    let mut data = (0..).map_while(|i|{
        stringmaps.contigs().get_index(i)
                        }).collect::<Vec<&str>>()
                        .into_par_iter()
                        .zip(sizes)
                        .map(|(chrom, size)| {

        let region = format!("{chrom}").parse().expect("failed to parse region");

        let mut bcf_r;
        { 
            
            bcf_r = bcf::Reader::new(f.clone());
            bcf_r.read_file_format().expect("failed to read format");
            let _r_header = bcf_r.read_header().expect("failed to read header");

        };

        let records = bcf_r.query(stringmaps.contigs(), &index, &region)
                           .expect("failed to query index");
        
        let capacity = if let Some(cap) = size {
            cap
        } else {
            100_000_000 as usize
        };
        
        //preallocate vectors to store data.
        let mut samples: Vec<Vec<Option<i8>>> = (0..num_samples).map(|_| {
            Vec::with_capacity(capacity) 
        }).collect();

        records.for_each(|record| {

            //3 header values then genotypes are pairs
            //get only the GT array, drop other data
            let record = record.expect("failed to read record");
            let n_samp = record.genotypes().len();
            
            let gt_data = record.genotypes().as_ref();

            let allele_num = gt_data[2] >> 4;
            
            if allele_num == 2 {

                gt_data[3..(n_samp * 2 + 3)]
                       .into_iter().map(|val| -> i8 {
                    //ignore phasing
                    //convert back from bcf to vcf
                    // -1 is a missing value
                    ((val >> 1) - 1) as i8
                }).collect::<Vec<i8>>()
                .chunks_exact(2)
                .map(|chunk| { 
    
                    if chunk.iter().any( |val| (val == &2) | (val == &3) ) { 
                        //if a dual/triple snp
                        //return filterable value
                        return 16i8 
                    } else { //combine good values
                        chunk.iter().sum()
                    }

                }).zip(samples.iter_mut())
                .for_each(|(read_value, sample)| {
 
                    let value = if read_value == -2 { 
                        None
                    } else {
                        Some(read_value)
                    };

                    sample.push(value);

                });

            }
           //if not 2 alleles do nothing

        });


        let data = samples.into_par_iter()
                .map(|sample| {
                    let chunked = sample.into_iter().collect();
                    chunked
                })
                .collect::<Vec<ChunkedArray<Int8Type>>>();
        
        data
    }).reduce(
        || (0..num_samples).map(|_| ChunkedArray::<Int8Type>::default())
                            .collect::<Vec<ChunkedArray<Int8Type>>>(),
        |a_s, b_s| {
            a_s.into_iter()
             .zip(b_s)
             .map(|(mut a, b)| {
                a.append(&b);
                a
             }).collect::<Vec<ChunkedArray<Int8Type>>>()
        }
    );             

    data = {
        let mask = data.iter() //create a mask of all records without dual snps
                .map(|array| array.not_equal(16))
                .reduce(|accum, array| accum & array)
                .unwrap();

        data.into_par_iter()
            .map(|array| {
                let array = array.filter(&mask).unwrap();
                array.rechunk()
            })
            .collect()
    };
    let data = Arc::new(data);

    //preinitiallize matrix with 0.0 
    let mut matrix = (0_usize..num_samples).map(|_| repeat(0.0).take(num_samples).collect())
                                            .collect::<Vec<Vec<f64>>>();

    
    //returns all combinations of all numbers in the sequence as tuples
    (0_usize..num_samples).tuple_combinations::<(usize, usize)>()
                        .collect::<Vec<(usize, usize)>>()
                        .into_par_iter()
                        .map(|(i, j)| {

                            (&data[i] - &data[j]).abs().mean().unwrap_or(f64::NAN) / 2.0

                        }).collect::<Vec<f64>>()
                        .into_iter()
                        .zip((0_usize..num_samples).tuple_combinations::<(usize, usize)>())
                        .for_each(|(avg, (i, j))| {
                            //store values in the matrix in both directions
                            matrix[j][i] = avg;
                            matrix[i][j] = avg;

                        });

    matrix.iter()
          .zip(sample_names.iter())
          .for_each(|(line, samp_name)| {
            let line = line.iter()
                           .join("\t");
            println!("{samp_name}\t{line}");
    });


    Ok(())
}
