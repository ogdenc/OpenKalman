/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Randomization functions.
 */

#ifndef OPENKALMAN_RANDOMIZE_HPP
#define OPENKALMAN_RANDOMIZE_HPP

namespace OpenKalman
{
  // ----------- //
  //  randomize  //
  // ----------- //

  namespace detail
  {
#ifndef __cpp_concepts
    template<typename T, typename = void, typename = void>
    struct is_std_dist : std::false_type {};

    template<typename T>
    struct is_std_dist<T, std::void_t<typename T::result_type>, std::void_t<typename T::param_type>> : std::true_type {};
#endif


    template<typename random_number_generator>
    struct RandomizeGenerator
    {
      static auto& get()
      {
        static std::random_device rd;
        static std::decay_t<random_number_generator> gen {rd()};
        return gen;
      }
    };


    template<typename random_number_generator, typename distribution_type>
    struct RandomizeOp
    {
      template<typename G, typename D>
      RandomizeOp(G& g, D&& d) : generator{g}, distribution{std::forward<D>(d)} {}

      auto operator()() const
      {
        if constexpr (std::is_arithmetic_v<distribution_type>)
          return distribution;
        else
          return distribution(generator);
      }

    private:

      std::decay_t<random_number_generator>& generator;
      mutable std::decay_t<distribution_type> distribution;
    };


    template<typename G, typename D>
    RandomizeOp(G&, D&&) -> RandomizeOp<G, D>;

  } // namespace detail


  /**
   * \brief Create an indexible object with random values selected from one or more random distributions.
   * \details This is essentially a specialized version of \ref n_ary_operation with the nullary operator
   * being a randomization function. The distributions are allocated to each element of the object, according to one
   * of the following options:
   *  - One distribution for all elements. The following example constructs a 2-by-2 matrix (m) in which each
   *  element is a random value selected based on a distribution with mean 1.0 and standard deviation 0.3:
   *   \code
   *     using N = std::normal_distribution<double>;
   *     using Mat = Eigen::Matrix<double, 2, 2>;
   *     auto g = std::mt19937 {};
   *     Mat m = randomize<Mat>(g, std::tuple {2, 2}, N {1.0, 0.3}));
   *   \endcode
   *  - One distribution for each element. The following code constructs a 2-by-2 matrix m containing
   *  random values around mean 1.0, 2.0, 3.0, and 4.0 (in row-major order), with standard deviations of
   *  0.3, 0.3, 0.0 (by default, since no s.d. is specified as a parameter), and 0.3:
   *   \code
   *     using N = std::normal_distribution<double>;
   *     auto m = randomize<Eigen::Matrix<double, 2, 2>, 0, 1>(g,
   *       std::tuple {Dimensions<2>{}, Dimensions<2>{}}, N {1.0, 0.3}, N {2.0, 0.3}, 3.0, N {4.0, 0.3})));
   *   \endcode
   *  - One distribution for each row. The following code constructs a 3-by-2 (n) or 2-by-2 (o) matrices
   *  in which elements in each row are selected according to the three (n) or two (o) listed distribution
   *  parameters:
   *   \code
   *     auto n = randomize<Eigen::Matrix<double, 3, 2>, 0>(g, std::tuple {Dimensions<3>{}, 2},
   *       N {1.0, 0.3}, 2.0, N {3.0, 0.3})));
   *     auto o = randomize<Eigen::Matrix<double, 2, 2>, 0>(g, std::tuple {Dimensions<2>{}, 2},
   *       N {1.0, 0.3}, N {2.0, 0.3})));
   *   \endcode
   *  - One distribution for each column. The following code constructs 2-by-3 matrix p
   *  in which elements in each column are selected according to the three listed distribution parameters:
   *   \code
   *     auto p = randomize<Eigen::Matrix<double, 2, 3>, 1>(g, std::tuple {2, Dimensions<3>{}},
   *       N {1.0, 0.3}, 2.0, N {3.0, 0.3})));
   *   \endcode
   * \tparam PatternMatrix An indexible object corresponding to the result type. Its dimensions need not match the
   * specified dimensions Ds
   * \tparam indices The indices, if any, for which there is a distinct distribution. If not provided, this can in some
   * cases be inferred from the number of Dists provided.
   * \tparam random_number_generator The random number generator (e.g., std::mt19937).
   * \tparam Ds \ref vector_space_descriptor objects for each index the result. They need not correspond to the dimensions of PatternMatrix.
   * \tparam Dists One or more distributions (e.g., std::normal_distribution<double>)
   * \sa n_ary_operation
   */
#ifdef __cpp_concepts
  template<indexible PatternMatrix, std::size_t...indices, std::uniform_random_bit_generator random_number_generator,
    vector_space_descriptor...Ds, typename...Dists>
  requires ((fixed_vector_space_descriptor<std::tuple_element_t<indices, std::tuple<Ds...>>>) and ...) and
    (sizeof...(Dists) == (1 * ... * dimension_size_of_v<std::tuple_element_t<indices, std::tuple<Ds...>>>)) and
    ((std::is_arithmetic_v<std::decay_t<Dists>> or
      requires { typename std::decay_t<Dists>::result_type; typename std::decay_t<Dists>::param_type; }) and ...)
#else
  template<typename PatternMatrix, std::size_t...indices, typename random_number_generator, typename...Ds,
    typename...Dists, std::enable_if_t<indexible<PatternMatrix> and (vector_space_descriptor<Ds> and ...) and
    ((fixed_vector_space_descriptor<std::tuple_element_t<indices, std::tuple<Ds...>>>) and ...) and
    (sizeof...(Dists) == (1 * ... * dimension_size_of<std::tuple_element_t<indices, std::tuple<Ds...>>>::value)) and
    ((std::is_arithmetic_v<std::decay_t<Dists>> or detail::is_std_dist<std::decay_t<Dists>>::value) and ...), int> = 0>
#endif
  constexpr auto
  randomize(random_number_generator& gen, const std::tuple<Ds...>& ds_tuple, Dists&&...dists)
  {
    auto ret {n_ary_operation<PatternMatrix, indices...>(ds_tuple, detail::RandomizeOp {gen, (std::forward<Dists>(dists))}...)};
    if constexpr (sizeof...(Dists) == 1) return to_dense_object(std::move(ret));
    else return ret;
  }


  /**
   * \overload
   * \brief Create an indexible object with random values, using std::mt19937 as the random number engine.
   */
#ifdef __cpp_concepts
  template<indexible PatternMatrix, std::size_t...indices, vector_space_descriptor...Ds, typename...Dists>
  requires ((fixed_vector_space_descriptor<std::tuple_element_t<indices, std::tuple<Ds...>>>) and ...) and
    (sizeof...(Dists) == (1 * ... * dimension_size_of_v<std::tuple_element_t<indices, std::tuple<Ds...>>>)) and
    ((std::is_arithmetic_v<std::decay_t<Dists>> or
      requires { typename std::decay_t<Dists>::result_type; typename std::decay_t<Dists>::param_type; }) and ...)
#else
  template<typename PatternMatrix, std::size_t...indices, typename...Ds, typename...Dists,
    std::enable_if_t<indexible<PatternMatrix> and (vector_space_descriptor<Ds> and ...) and
    ((fixed_vector_space_descriptor<std::tuple_element_t<indices, std::tuple<Ds...>>>) and ...) and
    (sizeof...(Dists) == (1 * ... * dimension_size_of<std::tuple_element_t<indices, std::tuple<Ds...>>>::value)) and
    ((std::is_arithmetic_v<std::decay_t<Dists>> or detail::is_std_dist<std::decay_t<Dists>>::value) and ...), int> = 0>
#endif
  constexpr auto
  randomize(const std::tuple<Ds...>& d_tuple, Dists&&...dists)
  {
    auto& gen = detail::RandomizeGenerator<std::mt19937>::get();
    return randomize<PatternMatrix, indices...>(gen, d_tuple, std::forward<Dists>(dists)...);
  }


  /**
   * \overload
   * \brief Fill a fixed-sized indexible object with random values selected from one or more random distributions.
   * \details The distributions are allocated to each element of the matrix, according to one of the following options:
   *  - One distribution for all matrix elements. The following example constructs a 2-by-2 matrix (m) in which each
   *  element is a random value selected based on a distribution with mean 1.0 and standard deviation 0.3:
   *   \code
   *     using N = std::normal_distribution<double>;
   *     using Mat = Eigen::Matrix<double, 2, 2>;
   *     auto g = std::mt19937 {};
   *     Mat m = randomize<Mat>(N {1.0, 0.3}));
   *   \endcode
   *  - One distribution for each matrix element. The following code constructs a 2-by-2 matrix m containing
   *  random values around mean 1.0, 2.0, 3.0, and 4.0 (in row-major order), with standard deviations of
   *  0.3, 0.3, 0.0 (by default, since no s.d. is specified as a parameter), and 0.3:
   *   \code
   *     using N = std::normal_distribution<double>;
   *     auto m = randomize<Eigen::Matrix<double, 2, 2>, 0, 1>(g, N {1.0, 0.3}, N {2.0, 0.3}, 3.0, N {4.0, 0.3})));
   *   \endcode
   *  - One distribution for each row. The following code constructs a 3-by-2 (n) or 2-by-2 (o) matrices
   *  in which elements in each row are selected according to the three (n) or two (o) listed distribution
   *  parameters:
   *   \code
   *     auto n = randomize<Eigen::Matrix<double, 3, 2>, 0>(g, N {1.0, 0.3}, 2.0, N {3.0, 0.3})));
   *     auto o = randomize<Eigen::Matrix<double, 2, 2>, 0>(g, N {1.0, 0.3}, N {2.0, 0.3})));
   *   \endcode
   *  - One distribution for each column. The following code constructs 2-by-3 matrix p
   *  in which elements in each column are selected according to the three listed distribution parameters:
   *   \code
   *     auto p = randomize<Eigen::Matrix<double, 2, 3>, 1>(g, N {1.0, 0.3}, 2.0, N {3.0, 0.3})));
   *   \endcode
   * \tparam PatternMatrix A fixed-size matrix
   * \tparam random_number_generator The random number generator (e.g., std::mt19937).
   */
#ifdef __cpp_concepts
  template<indexible PatternMatrix, std::size_t...indices, std::uniform_random_bit_generator random_number_generator,
    typename...Dists>
  requires (not has_dynamic_dimensions<PatternMatrix>) and
    (sizeof...(Dists) == (1 * ... * index_dimension_of_v<PatternMatrix, indices>)) and
    ((std::is_arithmetic_v<Dists> or requires { typename Dists::result_type; typename Dists::param_type; }) and ...)
#else
  template<typename PatternMatrix, std::size_t...indices, typename random_number_generator, typename...Dists,
    std::enable_if_t<indexible<PatternMatrix> and (not has_dynamic_dimensions<PatternMatrix>) and
    (sizeof...(Dists) == (1 * ... * index_dimension_of<PatternMatrix, indices>::value)) and
    ((std::is_arithmetic_v<Dists> or detail::is_std_dist<Dists>::value) and ...), int> = 0>
#endif
  constexpr auto
  randomize(random_number_generator& gen, Dists&&...dists)
  {
    auto d_tup = all_vector_space_descriptors<PatternMatrix>();
    return randomize<PatternMatrix, indices...>(gen, d_tup, std::forward<Dists>(dists)...);
  }


  /**
   * \overload
   * \brief Fill a fixed-sized indexible object with random values using std::mt19937 as the random number generator.
   */
#ifdef __cpp_concepts
  template<indexible PatternMatrix, std::size_t...indices, typename...Dists>
  requires (not has_dynamic_dimensions<PatternMatrix>) and
    (sizeof...(Dists) == (1 * ... * index_dimension_of_v<PatternMatrix, indices>)) and
    ((std::is_arithmetic_v<Dists> or requires { typename Dists::result_type; typename Dists::param_type; }) and ...)
#else
  template<typename PatternMatrix, std::size_t...indices, typename...Dists, std::enable_if_t<
    indexible<PatternMatrix> and (not has_dynamic_dimensions<PatternMatrix>) and
    (sizeof...(Dists) == (1 * ... * index_dimension_of<PatternMatrix, indices>::value)) and
    ((std::is_arithmetic_v<Dists> or detail::is_std_dist<Dists>::value) and ...), int> = 0>
#endif
  constexpr auto
  randomize(Dists&&...dists)
  {
    auto& gen = detail::RandomizeGenerator<std::mt19937>::get();
    return randomize<PatternMatrix, indices...>(gen, std::forward<Dists>(dists)...);
  }


} // namespace OpenKalman

#endif //OPENKALMAN_RANDOMIZE_HPP
