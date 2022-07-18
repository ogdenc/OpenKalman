/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for Eigen3::ZeroMatrix
 */

#ifndef OPENKALMAN_EIGEN3_ZEROMATRIX_HPP
#define OPENKALMAN_EIGEN3_ZEROMATRIX_HPP

#include <type_traits>

namespace OpenKalman::Eigen3
{

  // ZeroMatrix is declared in forward_class-declarations.hpp.

#ifdef __cpp_concepts
  template<indexible PatternMatrix>
#else
  template<typename PatternMatrix>
#endif
  struct ZeroMatrix : MatrixTraits<PatternMatrix>::template MatrixBaseFrom<ZeroMatrix<PatternMatrix>>
  {

  private:

    using nested_scalar = scalar_type_of_t<PatternMatrix>;
    static constexpr std::size_t max_indices = max_indices_of_v<PatternMatrix>;


    using MyDimensions = decltype(get_all_dimensions_of(std::declval<PatternMatrix>()));

  public:

    /**
     * \brief Construct a ZeroMatrix.
     * \details The constructor can take a number of \ref index_descriptor "index_descriptors" representing the
     * number of dimensions.
     */
#ifdef __cpp_concepts
    template<euclidean_index_descriptor...D> requires (sizeof...(D) == max_indices) and
      std::constructible_from<MyDimensions, D&&...>
#else
    template<typename...D,
      std::enable_if_t<(euclidean_index_descriptor<D> and ...) and sizeof...(D) == max_indices and
        std::is_constructible<MyDimensions, const D&...>::value, int> = 0>
#endif
    constexpr ZeroMatrix(D&&...d) : my_dimensions {std::forward<D>(d)...} {}

  private:

    template<std::size_t I>
    static constexpr auto make_dimensions_tuple()
    {
      if constexpr (I >= max_indices)
      {
        return std::tuple {};
      }
      else
      {
        using Dim = std::tuple_element_t<I, MyDimensions>;
        return std::tuple_cat(std::tuple {Dim {Dimensions<index_dimension_of_v<PatternMatrix, I>>{}}},
          make_dimensions_tuple<I + 1>());
      }
    }


    template<std::size_t I, typename N, typename...Ns>
    static constexpr auto make_dimensions_tuple(N n, Ns...ns)
    {
      using Dim = std::tuple_element_t<I, MyDimensions>;
      if constexpr (dynamic_dimension<PatternMatrix, I>)
        return std::tuple_cat(std::tuple {Dim {Dimensions{n}}}, make_dimensions_tuple<I + 1>(ns...));
      else
        return std::tuple_cat(std::tuple {Dim {Dimensions<index_dimension_of_v<PatternMatrix, I>>{}}},
          make_dimensions_tuple<I + 1>(n, ns...));
    }

  public:

    /**
     * \brief Construct a ZeroMatrix.
     * \details The constructor can take a number of arguments representing the number of dynamic dimensions.
     * For example, the following construct a 2-by-3 zero matrix:
     * \code
     * ZeroMatrix<Mdd>(2, 3) // Mdd has dynamic rows and columns.
     * ZeroMatrix<M2d>(3) // M2d has fixed rows and dynamic columns.
     * ZeroMatrix<Md3>(2) // Md2 has dynamic rows and fixed columns.
     * ZeroMatrix<M23>() // M23 has fixed rows and columns.
     * \endcode
     */
#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... I> requires (sizeof...(I) == number_of_dynamic_indices_v<PatternMatrix>)
#else
    template<typename...I, std::enable_if_t<(std::is_convertible<I, std::size_t>::value and ...) and
      (sizeof...(I) == number_of_dynamic_indices<PatternMatrix>::value), int> = 0>
#endif
    constexpr ZeroMatrix(const I...i)
      : my_dimensions {make_dimensions_tuple<0>(static_cast<const std::size_t>(i)...)} {}


    /**
     * \brief Copy constructor
     */
    constexpr ZeroMatrix(const ZeroMatrix&) = default;


    /**
     * \brief Move constructor
     */
    constexpr ZeroMatrix(ZeroMatrix&&) = default;

  private:

    template<typename T, std::size_t I>
    static constexpr bool arg_one_dim_matches()
    {
      constexpr auto dim1 = index_dimension_of_v<T, I>;
      constexpr auto dim2 = dimension_size_of_v<std::tuple_element_t<I, MyDimensions>>;
      return dim1 == dynamic_size or dim2 == dynamic_size or dim1 == dim2;
    };


    template<typename T, std::size_t...I>
    static constexpr bool arg_matches_impl(std::index_sequence<I...>)
    {
      return (arg_one_dim_matches<T, I>() and ...);
    };


#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct zero_arg_matches : std::false_type {};


#ifdef __cpp_concepts
    template<typename T> requires (arg_matches_impl<T>(std::make_index_sequence<max_indices_of_v<T>> {}))
    struct zero_arg_matches<T>
#else
    template<typename T>
    struct zero_arg_matches<T, std::enable_if_t<arg_matches_impl<T>(std::make_index_sequence<max_indices_of_v<T>> {})>>
#endif
      : std::true_type {};

  public:

    /**
     * \internal
     * \brief Construct a ZeroMatrix from another zero_matrix.
     * \tparam M A zero_matrix with a compatible shape.
     */
#ifdef __cpp_concepts
    template<zero_matrix M> requires (not std::same_as<M, ZeroMatrix>) and
      (max_indices_of_v<M> == max_indices_of_v<PatternMatrix>) and zero_arg_matches<M>::value
#else
    template<typename M, std::enable_if_t<zero_matrix<M> and (not std::is_same_v<M, ZeroMatrix>) and
      (max_indices_of_v<M> == max_indices_of_v<PatternMatrix>) and zero_arg_matches<M>::value, int> = 0>
#endif
    constexpr ZeroMatrix(M&& m) : my_dimensions {get_all_dimensions_of(std::forward<M>(m))} {}

  private:

    template<typename T, std::size_t...I>
    constexpr void check_runtime_sizes_impl(const T& t, std::index_sequence<I...>)
    {
      ([](const T& t, const auto& my_dims) {
        if constexpr (index_dimension_of_v<T, I> == dynamic_size)
        {
          constexpr auto t_dim = get_dimensions_of<I>(t);
          constexpr auto dim = get_dimension_size_of(std::get<I>(my_dims));
          if (t_dim != dim)
            throw std::logic_error {"In an argument to ZeroMatrix assignment operator, "
              "the dimension of index " + std::to_string(I) + " is " + std::to_string(t_dim) +
              ", which does not match index " + std::to_string(I) + " of the ZeroMatrix, which is " +
              std::to_string(dim)};;
        }
      }(t, my_dimensions), ...);
    };


    template<typename T>
    constexpr void check_runtime_sizes(const T& t)
    {
      if constexpr (has_dynamic_dimensions<T> or has_dynamic_dimensions<PatternMatrix>)
        return check_runtime_sizes_impl(t, std::make_index_sequence<max_indices_of_v<T>> {});
    };

  public:

    /**
     * \brief Copy assignment operator
     */
    constexpr ZeroMatrix& operator=(const ZeroMatrix& m)
    {
      check_runtime_sizes(m);
      return *this;
    }


    /**
     * \brief Move assignment operator
     */
    constexpr ZeroMatrix& operator=(ZeroMatrix&& m)
    {
      check_runtime_sizes(m);
      return *this;
    }


    /**
     * \internal
     * \brief Assign from another compatible zero_matrix.
     */
#ifdef __cpp_concepts
    template<zero_matrix M> requires (not std::same_as<M, ZeroMatrix>) and zero_arg_matches<M>::value
#else
    template<typename M, std::enable_if_t<zero_matrix<M> and (not std::is_same_v<M, ZeroMatrix>) and
      zero_arg_matches<M>::value, int> = 0>
#endif
    constexpr auto& operator=(M&& m)
    {
      check_runtime_sizes(m);
      return *this;
    }


    /**
     * \brief Element accessor.
     * \brief Element accessor.
     * \note Does not do any runtime bounds checking.
     * \param d The indices
     * \return The element corresponding to the indices (always zero).
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const std::size_t>...D> requires (sizeof...(D) <= max_indices_of_v<PatternMatrix>)
#else
    template<typename...D, std::enable_if_t<(std::is_convertible<D, const std::size_t>::value and ...) and
      (sizeof...(D) <= max_indices_of_v<PatternMatrix>), int> = 0>
#endif
    constexpr nested_scalar
    operator()(D...d) const
    {
      return 0;
    }


    /**
     * \brief Element accessor for a first-order tensor (e.g., a row or column vector).
     * \note Does not do any runtime bounds checking.
     * \param i The index.
     * \return The element at index i (always zero).
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const std::size_t> D> requires (tensor_order_of_v<PatternMatrix> <= 1)
#else
    template<typename D, std::enable_if_t<std::is_convertible<D, const std::size_t>::value and
      tensor_order_of<PatternMatrix>::value <= 1, int> = 0>
#endif
    constexpr nested_scalar
    operator[](D d) const
    {
      return 0;
    }

  private:

    MyDimensions my_dimensions;

#ifdef __cpp_concepts
    template<typename T, std::size_t N> friend struct interface::IndexTraits;
    template<typename T, std::size_t N> friend struct interface::CoordinateSystemTraits;
#else
    template<typename T, std::size_t N, typename Enable> friend struct interface::IndexTraits;
    template<typename T, std::size_t N, typename Enable> friend struct interface::CoordinateSystemTraits;
#endif

  };


  // ------------------------------- //
  //        Deduction guides         //
  // ------------------------------- //

#ifdef __cpp_concepts
  template<zero_matrix Arg>
#else
  template<typename Arg, std::enable_if_t<zero_matrix<Arg>, int> = 0>
#endif
  ZeroMatrix(Arg&&) -> ZeroMatrix<std::conditional_t<eigen_zero_expr<Arg> or eigen_constant_expr<Arg>,
    pattern_matrix_of_t<Arg>, std::decay_t<Arg>>>;


} // OpenKalman::Eigen3



#endif //OPENKALMAN_EIGEN3_ZEROMATRIX_HPP
