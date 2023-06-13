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
 * \brief Functions that concatenate other objects into a larger object.
 */

#ifndef OPENKALMAN_CONCATENATE_HPP
#define OPENKALMAN_CONCATENATE_HPP

namespace OpenKalman
{
  using namespace interface;

  // ============= //
  //  concatenate  //
  // ============= //

  namespace detail
  {
    template<typename T, typename U, std::size_t...indices, std::size_t...I>
    constexpr bool concatenate_dimensions_match_impl(std::index_sequence<I...>)
    {
      return (([](std::size_t i){ return ((i != indices) and ...); }(I) or
        dimension_size_of_index_is<T, I, index_dimension_of_v<U, I>, Likelihood::maybe>) and ...);
    }


    template<typename T, typename U, std::size_t...indices>
#ifdef __cpp_concepts
    concept concatenate_dimensions_match =
#else
    constexpr bool concatenate_dimensions_match =
#endif
      (concatenate_dimensions_match_impl<T, U, indices...>(std::make_index_sequence<max_indices_of_v<T>> {}));


    template<std::size_t I, std::size_t...indices, typename DTup, typename...DTups>
    constexpr decltype(auto) concatenate_index_descriptors_impl(DTup&& d_tup, DTups&&...d_tups)
    {
      if constexpr (((I == indices) or ...))
      {
        auto f = [](auto&& dtup){
          if constexpr (I >= std::tuple_size_v<std::decay_t<decltype(dtup)>>) return Dimensions<1> {};
          else return std::get<I>(std::forward<decltype(dtup)>(dtup));
        };
        return (f(std::forward<DTup>(d_tup)) + ... + f(std::forward<DTups>(d_tups)));
      }
      else
      {
        if constexpr (not (equivalent_to<std::tuple_element_t<I, DTup>, std::tuple_element_t<I, DTups>> and ...))
        {
          if (((std::get<I>(std::forward<DTup>(d_tup)) != std::get<I>(std::forward<DTups>(d_tups))) or ...))
            throw std::invalid_argument {"Arguments to concatenate do not match in at least index " + std::to_string(I)};
        }
        return std::get<I>(std::forward<DTup>(d_tup));
      }
    }


    template<std::size_t...indices, std::size_t...I, typename...DTups>
    constexpr decltype(auto) concatenate_index_descriptors(std::index_sequence<I...>, DTups&&...d_tups)
    {
      return std::tuple {concatenate_index_descriptors_impl<I, indices...>(std::forward<DTups>(d_tups)...)...};
    }


#ifdef __cpp_concepts
    template<typename T, typename...Ts>
    concept constant_concatenate_arguments =
      (constant_matrix<T, CompileTimeStatus::known> and ... and constant_matrix<Ts, CompileTimeStatus::known>) and
      (are_within_tolerance(constant_coefficient_v<T>, constant_coefficient_v<Ts>) and ...);
#else
    template<typename T, typename = void, typename...Ts>
    struct constant_concatenate_arguments_impl : std::false_type {};

    template<typename T, typename...Ts>
    struct constant_concatenate_arguments_impl<T,
      std::enable_if_t<(are_within_tolerance(constant_coefficient<T>::value, constant_coefficient<Ts>::value) and ...)>, Ts...>
      : std::true_type {};

    template<typename T, typename...Ts>
    constexpr bool constant_concatenate_arguments = constant_concatenate_arguments_impl<T, void, Ts...>::value;
#endif


    template<std::size_t index, std::size_t...indices, typename Args_tup, std::size_t...all_indices, std::size_t...pos>
    constexpr auto concatenate_diag_impl(Args_tup&& args_tup, std::index_sequence<all_indices...>, std::index_sequence<pos...>)
    {
      constexpr auto p_index = std::get<index>(std::tuple{pos...});
      if constexpr (((p_index == std::get<indices>(std::tuple{pos...})) and ...))
      {
        return std::get<p_index>(args_tup);
      }
      else
      {
        using Pattern = std::tuple_element_t<0, Args_tup>;
        return make_zero_matrix_like<Pattern>(get_dimensions_of<all_indices>(std::get<pos>(args_tup))...);
      }
    }


    template<std::size_t...indices, typename Ds_tup, typename Args_tup, std::size_t...all_indices, typename...Pos_seq>
    constexpr auto concatenate_diag(Ds_tup&& ds_tup, Args_tup&& args_tup, std::index_sequence<all_indices...> all_indices_seq, Pos_seq...pos_seq)
    {
      return tile(ds_tup, concatenate_diag_impl<indices...>(std::forward<Args_tup>(args_tup), all_indices_seq, pos_seq)...);
    }


    // all indices are processed. return the resulting indices of a particular block
    template<std::size_t...args_ix, std::size_t...pos>
    constexpr auto get_cat_indices(
      std::index_sequence<>,
      std::index_sequence<>,
      std::index_sequence<args_ix...>,
      std::index_sequence<pos...> pos_seq)
    {
      return std::tuple {pos_seq};
    }


    // all specified indices are examined, but there are still other ix indices to process
    template<std::size_t ix, std::size_t...ixs, std::size_t...args_ix, std::size_t...pos>
    constexpr auto get_cat_indices(
      std::index_sequence<ix, ixs...>,
      std::index_sequence<> index_seq,
      std::index_sequence<args_ix...> arg_ix_seq,
      std::index_sequence<pos...>)
    {
      return get_cat_indices(
        std::index_sequence<ixs...> {},
        index_seq,
        arg_ix_seq,
        std::index_sequence<pos..., 0> {}
        );
    }


    template<std::size_t ix, std::size_t...ixs, std::size_t index, std::size_t...indices, std::size_t...args_ix, std::size_t...pos>
    constexpr auto get_cat_indices(
      std::index_sequence<ix, ixs...> ix_seq,
      std::index_sequence<index, indices...> index_seq,
      std::index_sequence<args_ix...> arg_ix_seq,
      std::index_sequence<pos...> pos_seq)
    {
      if constexpr (ix == index) // Increase the rank of the resulting matrix.
      {
        static_assert (((index != indices) and ...), "No duplicate indices for concatenate function.");
        return std::tuple_cat(get_cat_indices(
          std::index_sequence<ixs...> {},
          std::index_sequence<indices...> {},
          arg_ix_seq,
          std::index_sequence<pos..., args_ix> {})...);
      }
      else if constexpr (((ix == indices) or ...)) // index, indices... are not in ascending order -- rotate and try again.
      {
        return get_cat_indices(
          ix_seq,
          std::index_sequence<indices..., index> {},
          arg_ix_seq,
          pos_seq);
      }
      else // index, indices... do not include ix.
      {
        return get_cat_indices(
          std::index_sequence<ixs...> {},
          index_seq,
          arg_ix_seq,
          std::index_sequence<pos..., 0> {});
      }
    }

  } // namespace detail


  /**
   * \brief Concatenate some number of math objects along one or more indices.
   * \tparam indices The indices along which the concatenation occurs. For example,
   *  - if indices is {0}, concatenation is along row index 0, and is a vertical concatenation;
   *  - if indices is {1}, concatenation is along column index 1, and is a horizontal concatenation; and
   *  - if indices is {0, 1} or {1, 0}, concatenation is diagonal along both row and column directions.
   * \tparam Arg First object to be concatenated
   * \tparam Args Other objects to be concatenated
   * \return The concatenated object
   */
#ifdef __cpp_concepts
  template<std::size_t...indices, indexible Arg, detail::concatenate_dimensions_match<Arg>...Args>
  requires (sizeof...(indices) > 0)
#else
  template<std::size_t...indices, typename Arg, typename...Args, std::enable_if_t<(sizeof...(indices) > 0) and
    (indexible<Arg> and ... and detail::concatenate_dimensions_match<Arg, Args>), int> = 0>
#endif
  constexpr decltype(auto)
  concatenate(Arg&& arg, Args&&...args)
  {
    auto seq = std::make_index_sequence<std::max({max_indices_of_v<Arg>, max_indices_of_v<Args>..., indices...})> {};
    auto d_tup = detail::concatenate_index_descriptors<indices...>(
      seq, get_all_dimensions_of(arg), get_all_dimensions_of(args)...);

    if constexpr (sizeof...(Args) == 0)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr ((zero_matrix<Arg> and ... and zero_matrix<Args>))
    {
      return std::apply([](auto&&...ds){ return make_zero_matrix_like<Arg>(std::forward<decltype(ds)>(ds)...); }, d_tup);
    }
    else if constexpr (sizeof...(indices) == 1 and detail::constant_concatenate_arguments<Arg, Args...>)
    {
      return std::apply([](auto&&...ds){
        return make_constant_matrix_like<Arg>(constant_coefficient<Arg>{}, std::forward<decltype(ds)>(ds)...);
      }, d_tup);
    }
    else if constexpr (sizeof...(indices) == 2 and ((indices == 0) or ...) and ((indices == 1) or ...) and
      (diagonal_matrix<Args> and ...))
    {
      return to_diagonal(concatenate<0>(diagonal_of(std::forward<Arg>(arg), std::forward<Args>(args)...)));
    }
    else if constexpr (sizeof...(indices) == 2 and ((indices == 0) or ...) and ((indices == 1) or ...) and
      (triangular_matrix<Arg> and ... and triangular_matrix<Args>) and
      ((triangular_matrix<Arg, TriangleType::upper> == triangular_matrix<Args, TriangleType::upper>) and ...))
    {
      return make_triangular_matrix<triangle_type_of_v<Arg>>(
        concatenate<0, 1>(nested_matrix(std::forward<Arg>(arg)), nested_matrix(std::forward<Args>(args))...));
    }
    else if constexpr (sizeof...(indices) == 2 and ((indices == 0) or ...) and ((indices == 1) or ...) and
      (hermitian_matrix<Arg> and ... and hermitian_matrix<Args>))
    {
      constexpr auto t = hermitian_adapter_type_of_v<Arg>;
      auto maybe_transpose = [](auto&& m) {
        using M = decltype(m);
        if constexpr(t == hermitian_adapter_type_of_v<M>) return nested_matrix(std::forward<M>(m));
        else return transpose(nested_matrix(std::forward<M>(m)));
      };
      return make_hermitian_matrix<t>(
        concatenate_diagonal(nested_matrix(std::forward<Arg>(arg)), maybe_transpose(std::forward<Args>(args))...));
    }
    else if constexpr (sizeof...(indices) == 1)
    {
      return tile(d_tup, std::forward<Arg>(arg), std::forward<Args>(args)...);
    }
    else
    {
      auto pos_tup = detail::get_cat_indices(seq, std::index_sequence<indices...> {},
        std::make_index_sequence<1 + sizeof...(Args)> {}, std::index_sequence<> {});

      return std::apply([&](auto...pos_seq){
        return detail::concatenate_diag<indices...>(d_tup,
          std::forward_as_tuple(std::forward<Arg>(arg), std::forward<Args>(args)...), seq, pos_seq...);
        }, pos_tup);
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_CONCATENATE_HPP
