/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief \ref reduce and related functions for reductions.
 */

#ifndef OPENKALMAN_REDUCTION_FUNCTIONS_HPP
#define OPENKALMAN_REDUCTION_FUNCTIONS_HPP

namespace OpenKalman
{
  using namespace interface;

  // -------- //
  //  reduce  //
  // -------- //

  namespace detail
  {
    template<std::size_t I, std::size_t...index, typename Arg>
    constexpr auto get_reduced_index(Arg&& arg)
    {
      if constexpr (((I == index) or ...))
      {
        using T = index_descriptor_of_t<Arg, I>;
        if constexpr (has_uniform_dimension_type<T>) return uniform_dimension_type_of_t<T>{};
        else return Dimensions<1>{};
      }
      else return get_index_descriptor<I>(std::forward<Arg>(arg));
    }


    template<std::size_t...index, typename C, typename T, std::size_t...I>
    constexpr auto make_constant_matrix_reduction(C&& c, T&& t, std::index_sequence<I...>)
    {
      return make_constant_matrix_like<T>(std::forward<C>(c), get_reduced_index<I, index...>(std::forward<T>(t))...);
    }


    template<std::size_t...indices, typename T, std::size_t...Is>
    constexpr auto count_reduced_dimensions(const T& t, std::index_sequence<Is...>)
    {
      if constexpr ((dynamic_dimension<T, indices> or ...))
      {
        return ([](const T& t){
          constexpr auto I = Is;
          if constexpr (((I == indices) or ...)) return get_index_dimension_of<I>(t);
          else return 1;
        }(t) * ... * 1);
      }
      else
      {
        constexpr auto dim = ([]{
          constexpr auto I = Is;
          if constexpr (((I == indices) or ...)) return index_dimension_of_v<T, I>;
          else return 1;
        }() * ... * 1);
        return std::integral_constant<std::size_t, dim>{};
      }
    }


#ifdef __cpp_concepts
    template<static_index_value Dim, typename BinaryOperation, typename Constant>
#else
    template<typename Dim, typename BinaryOperation, typename Constant, std::enable_if_t<static_index_value<Dim>, int> = 0>
#endif
    constexpr auto scalar_reduce_operation(Dim dim, const BinaryOperation& op, const Constant& c)
    {
      if constexpr (Dim::value <= 1) return c;
      else if constexpr (internal::is_plus<BinaryOperation>::value)
        return internal::scalar_constant_operation {std::multiplies<>{}, c, dim};
      else if constexpr (internal::is_multiplies<BinaryOperation>::value)
        return internal::scalar_constant_pow(c, dim);
      else return internal::scalar_constant_operation {op,
        scalar_reduce_operation(std::integral_constant<std::size_t, std::size_t{dim} - 1>{}, op, c), c};
    }


#ifdef __cpp_concepts
    template<dynamic_index_value Dim, typename BinaryOperation, typename Constant>
#else
    template<typename Dim, typename BinaryOperation, typename Constant, std::enable_if_t<dynamic_index_value<Dim>, int> = 0>
#endif
    constexpr auto scalar_reduce_operation(Dim dim, const BinaryOperation& op, const Constant& c)
    {
      if (dim <= 1) return get_scalar_constant_value(c);
      else if constexpr (internal::is_plus<BinaryOperation>::value) return get_scalar_constant_value(c) * dim;
      else if constexpr (internal::is_multiplies<BinaryOperation>::value) return std::pow(get_scalar_constant_value(c), dim);
      else return op(scalar_reduce_operation(dim - 1, op, c), get_scalar_constant_value(c));
    }



    template<typename Arg, std::size_t...I>
    constexpr bool has_uniform_reduction_indices(std::index_sequence<I...>)
    {
      return ((has_uniform_dimension_type<index_descriptor_of_t<Arg, I>> or dynamic_dimension<Arg, I>) and ...);
    }


    template<typename BinaryFunction, typename Arg, std::size_t...indices>
    constexpr scalar_type_of_t<Arg>
    reduce_all_indices(const BinaryFunction& b, Arg&& arg, std::index_sequence<indices...>)
    {
      if constexpr (zero_matrix<Arg> and (internal::is_plus<BinaryFunction>::value or internal::is_multiplies<BinaryFunction>::value))
      {
        return 0;
      }
      else if constexpr (constant_matrix<Arg>)
      {
        constexpr auto seq = std::make_index_sequence<max_indices_of_v<Arg>> {};
        auto dim = count_reduced_dimensions<indices...>(arg, seq);
        return scalar_reduce_operation(dim, b, constant_coefficient {std::forward<Arg>(arg)});
      }
      else
      {
        decltype(auto) red = interface::ArrayOperations<std::decay_t<Arg>>::template reduce<indices...>(b, std::forward<Arg>(arg));
        using Red = decltype(red);

        static_assert(scalar_type<Red> or one_by_one_matrix<Red, Likelihood::maybe>);

        if constexpr (scalar_type<Red>)
          return std::forward<Red>(red);
        else if constexpr (element_gettable<Red, sizeof...(indices)>)
          return get_element(std::forward<Red>(red), static_cast<decltype(indices)>(0)...);
        else
          return interface::LinearAlgebra<std::decay_t<Red>>::trace(std::forward<Red>(red));
      }
    }

  } // namespace detail


  /**
   * \brief Perform a partial reduction based on an associative binary function, across one or more indices.
   * \details The binary function must be associative. (This is not enforced, but the order of operation is undefined.)
   * \tparam index an index to be reduced. For example, if the index is 0, the result will have only one row.
   * If the index is 1, the result will have only one column.
   * \tparam indices Other indicesto be reduced. Because the binary function is associative, the order
   * of the indices does not matter.
   * \tparam BinaryFunction A binary function invocable with two values of type <code>scalar_type_of_t<Arg></code>.
   * It must be an associative function. Preferably, it should be a constexpr function, and even more preferably,
   * it should be a standard c++ function such as std::plus or std::multiplies.
   * \tparam Arg The tensor
   * \returns A vector or tensor with reduced dimensions.
   */
#ifdef __cpp_concepts
  template<std::size_t index, std::size_t...indices, typename BinaryFunction, indexible Arg> requires
    ((index < max_indices_of_v<Arg>) and ... and (indices < max_indices_of_v<Arg>)) and
    std::is_invocable_r_v<scalar_type_of_t<Arg>, BinaryFunction&&, scalar_type_of_t<Arg>, scalar_type_of_t<Arg>> and
    (detail::has_uniform_reduction_indices<Arg>(std::index_sequence<index, indices...> {}))
#else
  template<std::size_t index, std::size_t...indices, typename BinaryFunction, typename Arg, std::enable_if_t<
    indexible<Arg> and ((index < max_indices_of<Arg>::value) and ... and (indices < max_indices_of<Arg>::value)) and
    std::is_invocable_r<typename scalar_type_of<Arg>::type, BinaryFunction&&,
      typename scalar_type_of<Arg>::type, typename scalar_type_of<Arg>::type>::value and
    (detail::has_uniform_reduction_indices<Arg>(std::index_sequence<index, indices...> {})), int> = 0>
#endif
  constexpr auto
  reduce(const BinaryFunction& b, Arg&& arg)
  {
    constexpr auto max_indices = max_indices_of_v<Arg>;
    constexpr std::make_index_sequence<max_indices> seq;

    if constexpr (covariance<Arg>)
    {
      using RC = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<index_descriptor_of_t<Arg, 0>>, index_descriptor_of_t<Arg, 0>>;
      using CC = std::conditional_t<((index == 1) or ... or (indices == 1)),
        uniform_dimension_type_of_t<index_descriptor_of_t<Arg, 1>>, index_descriptor_of_t<Arg, 1>>;
      auto m = reduce<index, indices...>(b, to_covariance_nestable(std::forward<Arg>(arg)));
      return Matrix<RC, CC, decltype(m)> {std::move(m)};
    }
    else if constexpr(mean<Arg> and ((index != 0) or ... or (indices != 0)))
    {
      using C = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<index_descriptor_of_t<Arg, 0>>, index_descriptor_of_t<Arg, 0>>;
      auto m = from_euclidean<C>(reduce<index, indices...>(b, nested_matrix(to_euclidean(std::forward<Arg>(arg)))));
      return Mean<C, decltype(m)> {std::move(m)};
    }
    else if constexpr (euclidean_transformed<Arg> and ((index != 0) or ... or (indices != 0)))
    {
      using C = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<index_descriptor_of_t<Arg, 0>>, index_descriptor_of_t<Arg, 0>>;
      auto m = reduce<index, indices...>(b, nested_matrix(std::forward<Arg>(arg)));
      return EuclideanMean<C, decltype(m)> {std::move(m)};
    }
    else if constexpr (typed_matrix<Arg>)
    {
      using RC = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<index_descriptor_of_t<Arg, 0>>, index_descriptor_of_t<Arg, 0>>;
      using CC = std::conditional_t<((index == 1) or ... or (indices == 1)),
        uniform_dimension_type_of_t<index_descriptor_of_t<Arg, 1>>, index_descriptor_of_t<Arg, 1>>;
      auto m = reduce<index, indices...>(b, nested_matrix(std::forward<Arg>(arg)));
      return Matrix<RC, CC, decltype(m)> {std::move(m)};
    }
    else if constexpr (dimension_size_of_index_is<Arg, index, 1>)
    {
      if constexpr (sizeof...(indices) == 0) return std::forward<Arg>(arg);
      else return reduce<indices...>(b, std::forward<Arg>(arg));
    }
    else if constexpr (zero_matrix<Arg> and (internal::is_plus<BinaryFunction>::value or internal::is_multiplies<BinaryFunction>::value))
    {
      return detail::make_constant_matrix_reduction<index, indices...>(
        internal::ScalarConstant<Likelihood::definitely, scalar_type_of_t<Arg>, 0>{}, std::forward<Arg>(arg), seq);
    }
    else if constexpr (constant_matrix<Arg>)
    {
      auto dim = detail::count_reduced_dimensions<index, indices...>(arg, seq);
      auto c = detail::scalar_reduce_operation(dim, b, constant_coefficient{arg});
      return detail::make_constant_matrix_reduction<index, indices...>(std::move(c), std::forward<Arg>(arg), seq);
    }
    // \todo Add this after updating DiagonalMatrix to include diagonals of any rank
    //else if constexpr (constant_diagonal_matrix<Arg>)
    //{
    //}
    else
    {
      return interface::ArrayOperations<std::decay_t<Arg>>::template reduce<index, indices...>(b, std::forward<Arg>(arg));
    }
  }


  /**
   * \overload
   * \brief Perform a complete reduction based on an associative binary function, and return a scalar.
   * \details The binary function must be associative. (This is not enforced, but the order of operation is undefined.)
   * \tparam BinaryFunction A binary function invocable with two values of type <code>scalar_type_of_t<Arg></code>.
   * It must be an associative function. Preferably, it should be a constexpr function, and even more preferably,
   * it should be a standard c++ function such as std::plus or std::multiplies.
   * \tparam Arg The tensor
   * \returns A scalar representing a complete reduction.
   */
#ifdef __cpp_concepts
  template<typename BinaryFunction, indexible Arg> requires
    std::is_invocable_r_v<scalar_type_of_t<Arg>, BinaryFunction&&, scalar_type_of_t<Arg>, scalar_type_of_t<Arg>> and
    (detail::has_uniform_reduction_indices<Arg>(std::make_index_sequence<max_indices_of_v<Arg>> {}))
#else
  template<typename BinaryFunction, typename Arg, std::enable_if_t<indexible<Arg> and
    std::is_invocable_r<typename scalar_type_of<Arg>::type, BinaryFunction&&,
      typename scalar_type_of<Arg>::type, typename scalar_type_of<Arg>::type>::value and
    (detail::has_uniform_reduction_indices<Arg>(std::make_index_sequence<max_indices_of_v<Arg>> {})), int> = 0>
#endif
  constexpr scalar_type_of_t<Arg>
  reduce(const BinaryFunction& b, Arg&& arg)
  {
    return detail::reduce_all_indices(b, std::forward<Arg>(arg), std::make_index_sequence<max_indices_of_v<Arg>> {});
  }


  // ---------------- //
  //  average_reduce  //
  // ---------------- //

  /**
   * \brief Perform a partial reduction by taking the average along one or more indices.
   * \tparam index an index to be reduced. For example, if the index is 0, the result will have only one row.
   * If the index is 1, the result will have only one column.
   * \tparam indices Other indicesto be reduced. Because the binary function is associative, the order
   * of the indices does not matter.
   * \returns A vector or tensor with reduced dimensions.
   */
#ifdef __cpp_concepts
  template<std::size_t index, std::size_t...indices, indexible Arg> requires
    ((index < max_indices_of_v<Arg>) and ... and (indices < max_indices_of_v<Arg>)) and
    (detail::has_uniform_reduction_indices<Arg>(std::index_sequence<index, indices...> {}))
#else
  template<std::size_t index, std::size_t...indices, typename Arg, std::enable_if_t<indexible<Arg> and
    ((index < max_indices_of_v<Arg>) and ... and (indices < max_indices_of_v<Arg>)) and
    (detail::has_uniform_reduction_indices<Arg>(std::index_sequence<index, indices...> {})), int> = 0>
#endif
  constexpr auto
  average_reduce(Arg&& arg) noexcept
  {
    using Scalar = scalar_type_of_t<Arg>;
    constexpr auto max_indices = max_indices_of_v<Arg>;
    constexpr std::make_index_sequence<max_indices> seq;

    if constexpr (covariance<Arg>)
    {
      using RC = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<index_descriptor_of_t<Arg, 0>>, index_descriptor_of_t<Arg, 0>>;
      using CC = std::conditional_t<((index == 1) or ... or (indices == 1)),
        uniform_dimension_type_of_t<index_descriptor_of_t<Arg, 1>>, index_descriptor_of_t<Arg, 1>>;
      auto m = average_reduce<index, indices...>(to_covariance_nestable(std::forward<Arg>(arg)));
      return Matrix<RC, CC, decltype(m)> {std::move(m)};
    }
    else if constexpr(mean<Arg> and ((index != 0) or ... or (indices != 0)))
    {
      using C = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<index_descriptor_of_t<Arg, 0>>, index_descriptor_of_t<Arg, 0>>;
      auto m = from_euclidean<C>(average_reduce<index, indices...>(nested_matrix(to_euclidean(std::forward<Arg>(arg)))));
      return Mean<C, decltype(m)> {std::move(m)};
    }
    else if constexpr (euclidean_transformed<Arg> and ((index != 0) or ... or (indices != 0)))
    {
      using C = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<index_descriptor_of_t<Arg, 0>>, index_descriptor_of_t<Arg, 0>>;
      auto m = average_reduce<index, indices...>(nested_matrix(std::forward<Arg>(arg)));
      return EuclideanMean<C, decltype(m)> {std::move(m)};
    }
    else if constexpr (typed_matrix<Arg>)
    {
      using RC = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<index_descriptor_of_t<Arg, 0>>, index_descriptor_of_t<Arg, 0>>;
      using CC = std::conditional_t<((index == 1) or ... or (indices == 1)),
        uniform_dimension_type_of_t<index_descriptor_of_t<Arg, 1>>, index_descriptor_of_t<Arg, 1>>;
      auto m = average_reduce<index, indices...>(nested_matrix(std::forward<Arg>(arg)));
      return Matrix<RC, CC, decltype(m)> {std::move(m)};
    }
    else if constexpr (dimension_size_of_index_is<Arg, index, 1>)
    {
      if constexpr (sizeof...(indices) == 0) return std::forward<Arg>(arg);
      else return average_reduce<indices...>(std::forward<Arg>(arg));
    }
    else if constexpr (constant_matrix<Arg>)
    {
      return detail::make_constant_matrix_reduction<index, indices...>(constant_coefficient{arg}, std::forward<Arg>(arg), seq);
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      // \todo Handle diagonal tensors of order greater than 2 ?
      internal::scalar_constant_operation c {std::divides<>{}, constant_diagonal_coefficient{arg}, get_index_dimension_of<0>(arg)};
      auto ret = detail::make_constant_matrix_reduction<index>(std::move(c), std::forward<Arg>(arg), seq);
      if constexpr (sizeof...(indices) > 0) return average_reduce<indices...>(std::move(ret));
      else return ret;
    }
    else
    {
      return make_self_contained(reduce<index, indices...>(std::plus<Scalar> {}, std::forward<Arg>(arg)) /
        (get_index_dimension_of<index>(arg) * ... * get_index_dimension_of<indices>(arg)));
    }
  }


  namespace detail
  {
    template<typename Arg, std::size_t I, std::size_t...Is>
    constexpr std::size_t const_diagonal_matrix_dim(const Arg& arg, std::index_sequence<I, Is...>)
    {
      if constexpr (sizeof...(Is) == 0) return get_index_dimension_of<I>(arg);
      else if constexpr(not dynamic_dimension<Arg, I>) return index_dimension_of_v<Arg, I>;
      else return const_diagonal_matrix_dim(arg, std::index_sequence<Is...>{});
    }
  }


  /**
   * \overload
   * \brief Perform a complete reduction by taking the average along all indices and returning a scalar value.
   * \returns A scalar representing the average of all components.
   */
#ifdef __cpp_concepts
  template<indexible Arg> requires
    (detail::has_uniform_reduction_indices<Arg>(std::make_index_sequence<max_indices_of_v<Arg>> {}))
#else
  template<typename Arg, std::enable_if_t<indexible<Arg> and
    detail::has_uniform_reduction_indices<Arg>(std::make_index_sequence<max_indices_of_v<Arg>> {}), int> = 0>
#endif
  constexpr scalar_type_of_t<Arg>
  average_reduce(Arg&& arg) noexcept
  {
    if constexpr (zero_matrix<Arg>)
      return 0;
    else if constexpr (constant_matrix<Arg>)
      return constant_coefficient{arg}();
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      return static_cast<scalar_type_of_t<Arg>>(constant_diagonal_coefficient{arg}()) /
        detail::const_diagonal_matrix_dim(arg, std::make_index_sequence<max_indices_of_v<Arg>>{});
    }
    else
    {
      auto r = reduce(std::plus<scalar_type_of_t<Arg>> {}, std::forward<Arg>(arg));
      return r / (std::apply([](const auto&...d){ return (get_dimension_size_of(d) * ...); }, get_all_dimensions_of(arg)));
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_REDUCTION_FUNCTIONS_HPP
