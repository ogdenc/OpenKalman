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
 * \brief Overloaded general functions relating to subsets of math objects.
 */

#ifndef OPENKALMAN_SUBSET_FUNCTIONS_HPP
#define OPENKALMAN_SUBSET_FUNCTIONS_HPP

namespace OpenKalman
{
  using namespace interface;

  // ================= //
  //  Diagonal access  //
  // ================= //

  /**
   * \brief Extract the diagonal from a square matrix.
   * \tparam Arg A diagonal matrix
   * \returns Arg A column vector
   */
#ifdef __cpp_concepts
  template<typename Arg> requires (has_dynamic_dimensions<Arg> or square_matrix<Arg>)
#else
  template<typename Arg, std::enable_if_t<has_dynamic_dimensions<Arg> or square_matrix<Arg>, int> = 0>
#endif
  inline decltype(auto)
  diagonal_of(Arg&& arg)
  {
    using Scalar = scalar_type_of_t<Arg>;

    auto dim = get_dimensions_of<dynamic_rows<Arg> ? 1 : 0>(arg);

    if constexpr (one_by_one_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (identity_matrix<Arg>)
    {
      return make_constant_matrix_like<Arg, 1>(dim, Dimensions<1>{});
    }
    else if constexpr (zero_matrix<Arg>)
    {
      if constexpr (not square_matrix<Arg>) detail::check_if_square_at_runtime(arg);
      return make_zero_matrix_like<Arg>(dim, Dimensions<1>{});
    }
    else if constexpr (constant_matrix<Arg> or constant_diagonal_matrix<Arg>)
    {
      if constexpr (not constant_diagonal_matrix<Arg> and not square_matrix<Arg>)
        detail::check_if_square_at_runtime(arg);

      constexpr auto c = []{
        if constexpr (constant_matrix<Arg>) return constant_coefficient_v<Arg>;
        else return constant_diagonal_coefficient_v<Arg>;
      }();

#  if __cpp_nontype_template_args >= 201911L
      return make_constant_matrix_like<Arg, c>(dim, Dimensions<1>{});
#  else
      constexpr auto c_integral = static_cast<std::intmax_t>(c);
      if constexpr (are_within_tolerance(c, static_cast<Scalar>(c_integral)))
        return make_constant_matrix_like<Arg, c_integral>(dim, Dimensions<1>{});
      else
        return make_self_contained(c * to_native_matrix<Arg>(make_constant_matrix_like<Arg, 1>(dim, Dimensions<1>{})));
#  endif
    }
    else
    {
      return interface::Conversions<std::decay_t<Arg>>::diagonal_of(std::forward<Arg>(arg));
    }
  }


  // ============== //
  //  Block access  //
  // ============== //

  namespace detail
  {
    template<std::size_t min, std::size_t max>
    constexpr bool valid_chip_indices() { return true; }

    template<std::size_t min, std::size_t max, std::size_t index, std::size_t...indices>
    constexpr bool valid_chip_indices()
    {
      if constexpr (index < min or max <= index) return false;
      else return valid_chip_indices<index + 1, max, indices...>();
    }


    template<std::size_t I, std::size_t...index, typename Arg>
    constexpr auto get_chip_index_descriptor(Arg&& arg)
    {
      if constexpr (((I == index) or ...)) return Dimensions<1>{};
      else return get_dimensions_of<I>(std::forward<Arg>(arg));
    }


    template<std::size_t...index, typename Arg, std::size_t...I>
    constexpr auto make_zero_matrix_chip(Arg&& arg, std::index_sequence<I...>)
    {
      return make_zero_matrix_like<Arg>(get_chip_index_descriptor<I, index...>(std::forward<Arg>(arg))...);
    }


    template<auto constant, std::size_t...index, typename Arg, std::size_t...I>
    constexpr auto make_constant_matrix_chip(Arg&& arg, std::index_sequence<I...>)
    {
      return make_constant_matrix_like<Arg, constant>(get_chip_index_descriptor<I, index...>(std::forward<Arg>(arg))...);
    }
  }


  /**
   * \brief Extract a sub-array having rank less than the rank of the input object.
   * \details A chip is a special type of "thin" slice of width 1 in one or more dimensions, and otherwise no
   * reduction in extents. For example, the result could be a row vector, a column vector, a matrix (e.g., if the
   * input object is a rank-3 or higher tensor), etc.
   * \tparam indices The index or indices of the sliced dimension(s) in strictly ascending numerical order.
   * For example, if the input object is a matrix, a value of {0} will result in a row vector and a value of {1} will
   * result in a column vector. If the input object is a rank-3 tensor, a value of {0, 1} will result in a matrix.
   * \tparam index_values the index values corresponding to <code>indices</code>, in the same order. The values
   * may be positive \ref std::integral types or a positive \ref std::integral_constant.
   * \return A sub-array
   */
#ifdef __cpp_concepts
  template<std::size_t...indices, indexible Arg, std::convertible_to<std::size_t>...index_values>
  requires (sizeof...(indices) == sizeof...(index_values)) or
    (sizeof...(index_values) == 0 and ((index_dimension_of_v<Arg, indices> == 1) and ...))
#else
  template<std::size_t...indices, typename Arg, typename...index_values, std::enable_if_t<indexible<Arg> and
    (std::is_convertible<index_values, std::size_t>::value and ...) and
    ((sizeof...(indices) == sizeof...(index_values)) or
      (sizeof...(index_values) == 0 and ((index_dimension_of_v<Arg, indices> == 1) and ...))), int> = 0>
#endif
  constexpr decltype(auto) chip(Arg&& arg, index_values...is)
  {
    constexpr std::size_t max_indices = max_indices_of_v<Arg>;
    static_assert(detail::valid_chip_indices<0, max_indices, indices...>() == sizeof...(indices),
      "Chip indices must be unique and in ascending order");

    if constexpr (((index_dimension_of_v<Arg, indices> == 1) and ...))
    {
      if constexpr (sizeof...(is) > 0)
      {
        if (std::max({is...}) > 0) throw std::out_of_range {
          ("All chip row indices must be 0, but at least one is out of range: " + ... +
          ("ix_" + std::to_string(indices) + "==" + std::to_string(is) + " "))};
      }
      return std::forward<Arg>(arg);
    }
    else if constexpr (zero_matrix<Arg>)
    {
      constexpr auto seq = std::make_index_sequence<max_indices> {};
      return detail::make_zero_matrix_chip<indices...>(std::forward<Arg>(arg), seq);
    }
    else if constexpr (constant_matrix<Arg>)
    {
      constexpr auto c = constant_coefficient_v<Arg>;
      constexpr auto seq = std::make_index_sequence<max_indices> {};
      return detail::make_constant_matrix_chip<c, indices...>(std::forward<Arg>(arg), seq);
    }
    else
    {
      return interface::Subsets<std::decay_t<Arg>>::template chip<indices...>(std::forward<Arg>(arg), is...);
    }
  }


  /**
   * \brief Extract one column from a matrix or other tensor.
   * \details The index of the column may be specified at either compile time <em>or</em> at runtime, but not both.
   * \tparam compile_time_index The index of the column, if specified at compile time
   * \tparam Arg The matrix or other tensor from which the column is to be extracted
   * \tparam runtime_index_t The type of the index of the column, if the index is specified at runtime. This type
   * should be convertible to <code>std::size_t</code>
   * \return A \ref column_vector
   * \todo remove as redundant?
   */
#ifdef __cpp_concepts
  template<std::size_t...compile_time_index, typename Arg, std::convertible_to<const std::size_t>...runtime_index_t> requires
    (sizeof...(compile_time_index) + sizeof...(runtime_index_t) == 1) and
    (dynamic_columns<Arg> or ((compile_time_index + ... + 0) < column_dimension_of_v<Arg>))
#else
  template<std::size_t...compile_time_index, typename Arg, typename...runtime_index_t, std::enable_if_t<
    (std::is_convertible_v<runtime_index_t, const std::size_t> and ...) and
    (sizeof...(compile_time_index) + sizeof...(runtime_index_t) == 1) and
    (dynamic_columns<Arg> or ((compile_time_index + ... + 0) < column_dimension_of<Arg>::value)), int> = 0>
#endif
  constexpr decltype(auto)
  column(Arg&& arg, runtime_index_t...i)
  {
    return chip<1>(std::forward<Arg>(arg), std::integral_constant<std::size_t, compile_time_index> {}..., i...);
  }


  /**
   * \brief Extract one row from a matrix or other tensor.
   * \details The index of the row may be specified at either compile time <em>or</em> at runtime, but not both.
   * \tparam compile_time_index The index of the row, if specified at compile time
   * \tparam Arg The matrix or other tensor from which the row is to be extracted
   * \tparam runtime_index_t The type of the index of the row, if the index is specified at runtime. This type
   * should be convertible to <code>std::size_t</code>
   * \return A \ref row_vector
   * \todo remove as redundant?
   */
#ifdef __cpp_concepts
  template<std::size_t...compile_time_index, typename Arg, std::convertible_to<const std::size_t>...runtime_index_t> requires
    (sizeof...(compile_time_index) + sizeof...(runtime_index_t) == 1) and
    (dynamic_rows<Arg> or ((compile_time_index + ... + 0) < row_dimension_of_v<Arg>))
#else
  template<size_t...compile_time_index, typename Arg, typename...runtime_index_t, std::enable_if_t<
    (std::is_convertible_v<runtime_index_t, const std::size_t> and ...) and
    (sizeof...(compile_time_index) + sizeof...(runtime_index_t) == 1) and
    (dynamic_rows<Arg> or ((compile_time_index + ... + 0) < row_dimension_of<Arg>::value)), int> = 0>
#endif
  constexpr decltype(auto)
  row(Arg&& arg, runtime_index_t...i)
  {
    return chip<0>(std::forward<Arg>(arg), std::integral_constant<std::size_t, compile_time_index> {}..., i...);
  }


  // ============= //
  //  concatenate  //
  // ============= //

  namespace detail
  {
    template<typename T, typename U, std::size_t...indices, std::size_t...I>
    constexpr bool concatenate_dimensions_match_impl(std::index_sequence<I...>)
    {
      return (([](std::size_t i){ return ((i != indices) and ...); }(I) or dynamic_dimension<T, I> or
        dynamic_dimension<U, I> or index_dimension_of_v<T, I> == index_dimension_of_v<U, I>) and ...);
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
          if constexpr (I >= std::tuple_size_v<decltype(dtup)>) return Dimensions<1> {};
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
    concept constant_concatenate_arguments = (constant_matrix<T> and ... and constant_matrix<Ts>) and
      (are_within_tolerance(constant_coefficient_v<T>, constant_coefficient_v<Ts>) and ...);
#else
    template<typename T, typename = void, typename...Ts>
    struct constant_concatenate_arguments_impl : std::false_type {};

    template<typename T, typename...Ts>
    struct constant_concatenate_arguments_impl<T,
      std::enable_if_t<(are_within_tolerance(constant_coefficient_v<T>, constant_coefficient_v<Ts>) and ...)>, Ts...>
      : std::true_type {};

    template<typename T, typename...Ts>
    constexpr bool constant_concatenate_arguments = constant_concatenate_arguments_impl<T, void, Ts...>::value;
#endif

  } // namespace detail


  /**
   * \brief Concatenate some number of math objects along one or more indices.
   * \tparam indices The indices along which the concatenation occurs. For example,
   *  - if indices is {0}, concatenation is along row index 0, and is a vertical concatenation;
   *  - if indices is {1}, concatenation is along column index 1, and is a horizontal concatenation; and
   *  - if indices is {0, 1}, concatenation is diagonal along both row and column directions.
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
      constexpr auto c = constant_coefficient_v<Arg>;
      return std::apply([](auto&&...ds){ return make_constant_matrix_like<Arg, c>(std::forward<decltype(ds)>(ds)...); }, d_tup);
    }
    else if constexpr (sizeof...(indices) == 2 and ((indices == 0) or ...) and ((indices == 1) or ...) and
      (diagonal_matrix<Args> and ...))
    {
      return to_diagonal(concatenate<0>(diagonal_of(std::forward<Arg>(arg), std::forward<Args>(args)...)));
    }
    else if constexpr (sizeof...(indices) == 2 and ((indices == 0) or ...) and ((indices == 1) or ...) and
      (triangular_matrix<Arg> and ... and triangular_matrix<Args>) and
      ((upper_triangular_matrix<Arg> == upper_triangular_matrix<Args>) and ...))
    {
      // \todo replace with new make_triangular_adapter function.
      return MatrixTraits<Arg>::make(
        concatenate<0, 1>(nested_matrix(std::forward<Arg>(arg)), nested_matrix(std::forward<Args>(args))...));
    }
    else if constexpr (sizeof...(indices) == 2 and ((indices == 0) or ...) and ((indices == 1) or ...) and
      (self_adjoint_matrix<Arg> and ... and self_adjoint_matrix<Args>))
    {
      constexpr auto t = self_adjoint_triangle_type_of_v<Arg>;
      auto maybe_transpose = [](auto&& m) {
        using M = decltype(m);
        if constexpr(t == self_adjoint_triangle_type_of_v<M>) return nested_matrix(std::forward<M>(m));
        else return transpose(nested_matrix(std::forward<M>(m)));
      };
      // \todo replace with new make_hermitian_adapter function.
      return MatrixTraits<Arg>::make(
        concatenate_diagonal(nested_matrix(std::forward<Arg>(arg)), maybe_transpose(std::forward<Args>(args))...));
    }
    else
    {
      auto m = interface::Subsets<std::decay_t<Arg>>::template concatenate<indices...>(
        to_native_matrix(std::forward<Args>(args))...);
      if constexpr ((has_any_typed_index<Args> or ...))
      {
        // \todo create Matrix object using a new make_matrix function.
        return m;
      }
      else
      {
        return m;
      }
    }
  }


  // ======= //
  //  split  //
  // ======= //

  /*namespace detail
  {
    template<std::size_t...Is, typename D, typename DTup>
    constexpr bool split_dimensions_match_impl(std::index_sequence<0, Is...>, const D& d, const DTup& d_tup)
    {
      return (std::get<0>(d_tup) + ... + std::get<Is>(d_tup)) == d;
    }


    template<typename D, typename IndexDescriptorTuple>
#ifdef __cpp_concepts
    concept split_dimensions_match = dynamic_index_descriptor<D> or
      requires(const D& d, const IndexDescriptorTuple& d_tup){
        split_dimensions_match_impl(std::make_index_sequence<std::tuple_size_v<IndexDescriptorTuple>> {}, d, d_tup);
      };
#else
    constexpr bool split_dimensions_match = dynamic_index_descriptor<D> or
      split_dimensions_match_impl(std::make_index_sequence<std::tuple_size_v<IndexDescriptorTuple>> {}, d, d_tup);
#endif

  }*/ // namespace detail


/*#ifdef __cpp_concepts
  template<std::size_t...indices, indexible Arg, typename...IndexDescriptorTuples>
  requires (sizeof...(indices) == sizeof...(IndexDescriptorTuples)) and
    (detail::split_dimensions_match<coefficient_types_of_t<Arg, indices>, IndexDescriptorTuples> and ...)
#else
  template<typename F, bool euclidean, typename RC, typename...RCs, typename Arg, std::enable_if_t<eigen_matrix<Arg> and
    (dynamic_rows<Arg> or ((euclidean ? euclidean_dimension_size_of_v<RC> : dimension_size_of_v<RC>) + ... +
      (euclidean ? euclidean_dimension_size_of_v<RCs> : dimension_size_of_v<RCs>)) <= row_dimension_of<Arg>::value), int> = 0>
#endif
  inline auto
  split(Arg&& arg, const IndexDescriptorTuples&...d_tuples) noexcept
  {
    //auto seq = std::make_index_sequence<std::max({max_indices_of_v<Arg>, max_indices_of_v<Args>..., indices...})> {};
    //auto d_tup = detail::concatenate_index_descriptors<indices...>(
    //  seq, get_all_dimensions_of(arg), get_all_dimensions_of(args)...);


  }*/


  /*namespace detail
  {
    /// Make a tuple containing an Eigen matrix (general case).
    template<typename F, typename RC, typename CC, typename Arg>
    auto
    make_split_tuple(Arg&& arg)
    {
      auto val = F::template call<RC, CC>(std::forward<Arg>(arg));
      return std::tuple<const decltype(val)> {std::move(val)};
    }


    /// Make a tuple containing an Eigen::Block.
    template<typename F, typename RC, typename CC, typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
    auto
    make_split_tuple(Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>&& arg)
    {
      auto b = [](auto& arg) {
        using NonConstBlock = Eigen::Block<std::remove_const_t<XprType>, BlockRows, BlockCols, InnerPanel>;

        // A const_cast is necessary, because a const Eigen::Block cannot be inserted into a tuple.
        auto& xpr = const_cast<std::remove_const_t<XprType>&>(arg.nestedExpression());

        if constexpr (BlockRows == Eigen::Dynamic or BlockCols == Eigen::Dynamic)
          return NonConstBlock(xpr, arg.startRow(), arg.startCol(), get_dimensions_of<0>(arg), get_dimensions_of<1>(arg));
        else
          return NonConstBlock(xpr, arg.startRow(), arg.startCol());
      } (arg);

      auto val = F::template call<RC, CC>(std::move(b));
      return std::tuple<const decltype(val)> {std::move(val)};
    }

  }*/


  /**
   * \brief Split a matrix vertically.
   * \tparam F A class with a static <code>call</code> member to which the result is applied before creating the tuple.
   * \tparam euclidean Whether coefficients RC and RCs are transformed to Euclidean space.
   * \tparam RC TypedIndex for the first cut.
   * \tparam RCs TypedIndex for each of the second and subsequent cuts.
   * \todo add runtime-specified cuts
   */
/*#ifdef __cpp_concepts
  template<typename F, bool euclidean, typename RC, typename...RCs, eigen_matrix Arg>
  requires (dynamic_rows<Arg> or ((euclidean ? euclidean_dimension_size_of_v<RC> : dimension_size_of_v<RC>) + ... +
    (euclidean ? euclidean_dimension_size_of_v<RCs> : dimension_size_of_v<RCs>)) <= row_dimension_of_v<Arg>)
#else
  template<typename F, bool euclidean, typename RC, typename...RCs, typename Arg, std::enable_if_t<eigen_matrix<Arg> and
    (dynamic_rows<Arg> or ((euclidean ? euclidean_dimension_size_of_v<RC> : dimension_size_of_v<RC>) + ... +
      (euclidean ? euclidean_dimension_size_of_v<RCs> : dimension_size_of_v<RCs>)) <= row_dimension_of<Arg>::value), int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    // \todo Can g be replaced by make_self_contained?
    constexpr auto g = [](auto&& m) -> decltype(auto) {
      if constexpr (std::is_lvalue_reference_v<Arg>)
        return std::forward<decltype(m)>(m);
      else
        return make_dense_writable_matrix_from(std::forward<decltype(m)>(m));
    };

    using CC = Dimensions<dynamic_columns<Arg> ? 0 : column_dimension_of_v<Arg>>; // \todo fix this
    constexpr Eigen::Index dim1 = euclidean ? euclidean_dimension_size_of_v<RC> : dimension_size_of_v<RC>;

    if constexpr (sizeof...(RCs) > 0)
    {
      if constexpr (dynamic_rows<Arg>)
      {
        Eigen::Index dim2 = get_dimensions_of<0>(arg) - dim1;
        return std::tuple_cat(
          detail::make_split_tuple<F, RC, CC>(g(arg.topRows(dim1))),
          split_vertical<F, euclidean, RCs...>(g(arg.bottomRows(dim2))));
      }
      else
      {
        constexpr Eigen::Index dim2 = row_dimension_of_v<Arg> - dim1;
        return std::tuple_cat(
          detail::make_split_tuple<F, RC, CC>(g(arg.template topRows<dim1>())),
          split_vertical<F, euclidean, RCs...>(g(arg.template bottomRows<dim2>())));
      }
    }
    else if constexpr (dim1 < row_dimension_of_v<Arg>)
    {
      if constexpr (dynamic_rows<Arg>)
        return detail::make_split_tuple<F, RC, CC>(g(arg.topRows(dim1)));
      else
        return detail::make_split_tuple<F, RC, CC>(g(arg.template topRows<dim1>()));
    }
    else
    {
      return detail::make_split_tuple<F, RC, CC>(std::forward<Arg>(arg));
    }
  }*/


  /**
   * \brief Split a matrix vertically (case in which there is no split).
   */
/*#ifdef __cpp_concepts
  template<typename F = OpenKalman::internal::default_split_function, bool euclidean = false, eigen_matrix Arg>
#else
  template<typename F = OpenKalman::internal::default_split_function, bool euclidean = false,
    typename Arg, std::enable_if_t<eigen_matrix<Arg>, int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    return std::tuple {};
  }*/


  /**
   * \brief Split a matrix vertically.
   * \tparam F A class with a static <code>call</code> member to which the result is applied before creating the tuple.
   * \tparam RCs TypedIndex for each of the cuts.
   */
/*#ifdef __cpp_concepts
  template<typename F, fixed_index_descriptor RC, fixed_index_descriptor...RCs, eigen_matrix Arg>
  requires (not fixed_index_descriptor<F>) and
  (dynamic_rows<Arg> or (dimension_size_of_v<RC> + ... + dimension_size_of_v<RCs>) <= row_dimension_of_v<Arg>)
#else
  template<typename F, typename RC, typename...RCs, typename Arg, std::enable_if_t<eigen_matrix<Arg> and
    not fixed_index_descriptor<F> and (fixed_index_descriptor<RC> and ... and fixed_index_descriptor<RCs>) and
      (dynamic_rows<Arg> or (dimension_size_of_v<RC> + ... + dimension_size_of_v<RCs>) <= row_dimension_of<Arg>::value), int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    return split_vertical<F, false, RC, RCs...>(std::forward<Arg>(arg));
  }*/


  /**
   * \brief Split a matrix vertically.
   * \tparam RC TypedIndex for the first cut.
   * \tparam RCs TypedIndex for the second and subsequent cuts.
   */
/*#ifdef __cpp_concepts
  template<fixed_index_descriptor RC, fixed_index_descriptor...RCs, eigen_matrix Arg>
  requires (dynamic_rows<Arg> or (dimension_size_of_v<RC> + ... + dimension_size_of_v<RCs>) <= row_dimension_of_v<Arg>)
#else
  template<typename RC, typename...RCs, typename Arg, std::enable_if_t<eigen_matrix<Arg> and
    (fixed_index_descriptor<RC> and ... and fixed_index_descriptor<RCs>) and
    (dynamic_rows<Arg> or (dimension_size_of_v<RC> + ... + dimension_size_of_v<RCs>) <= row_dimension_of<Arg>::value), int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    return split_vertical<OpenKalman::internal::default_split_function, RC, RCs...>(std::forward<Arg>(arg));
  }*/


  /**
   * \brief Split a matrix vertically.
   * \tparam cut Number of rows in the first cut.
   * \tparam cuts Numbers of rows in the second and subsequent cuts.
   */
/*#ifdef __cpp_concepts
  template<std::size_t cut, std::size_t ... cuts, eigen_matrix Arg>
  requires (dynamic_rows<Arg> or (cut + ... + cuts) <= row_dimension_of_v<Arg>)
#else
  template<std::size_t cut, std::size_t ... cuts, typename Arg,
    std::enable_if_t<eigen_matrix<Arg> and
      (dynamic_rows<Arg> or (cut + ... + cuts) <= row_dimension_of<Arg>::value), int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    return split_vertical<Dimensions<cut>, Dimensions<cuts>...>(std::forward<Arg>(arg));
  }*/


  /**
   * \brief Split a matrix horizontally and invoke function F on each segment, returning a tuple.
   * \tparam CC TypedIndex for the first cut.
   * \tparam CCs TypedIndex for each of the second and subsequent cuts.
   * \tparam F An object having a static call() method to which the result is applied before creating the tuple.
   */
/*#ifdef __cpp_concepts
  template<typename F, fixed_index_descriptor CC, fixed_index_descriptor...CCs, eigen_matrix Arg>
  requires (not fixed_index_descriptor<F>) and
    (dynamic_columns<Arg> or (dimension_size_of_v<CC> + ... + dimension_size_of_v<CCs>) <= column_dimension_of_v<Arg>)
#else
  template<typename F, typename CC, typename...CCs, typename Arg,
    std::enable_if_t<eigen_matrix<Arg> and not fixed_index_descriptor<F> and
      (dynamic_columns<Arg> or (dimension_size_of_v<CC> + ... + dimension_size_of_v<CCs>) <= column_dimension_of<Arg>::value), int> = 0>
#endif
  inline auto
  split_horizontal(Arg&& arg) noexcept
  {
    // \todo Can g be replaced by make_self_contained?
    constexpr auto g = [](auto&& m) -> decltype(auto) {
      if constexpr (std::is_lvalue_reference_v<Arg&&>)
        return std::forward<decltype(m)>(m);
      else
        return make_dense_writable_matrix_from(std::forward<decltype(m)>(m));
    };

    using RC = Dimensions<dynamic_rows<Arg> ? 0 : row_dimension_of_v<Arg>>; // \todo fix this
    constexpr Eigen::Index dim1 = dimension_size_of_v<CC>;

    if constexpr(sizeof...(CCs) > 0)
    {
      if constexpr (dynamic_columns<Arg>)
      {
        Eigen::Index dim2 = get_dimensions_of<1>(arg) - dim1;
        return std::tuple_cat(
          detail::make_split_tuple<F, RC, CC>(g(arg.leftCols(dim1))),
          split_horizontal<F, CCs...>(g(arg.rightCols(dim2))));
      }
      else
      {
        constexpr Eigen::Index dim2 = column_dimension_of_v<Arg> - dim1;
        return std::tuple_cat(
          detail::make_split_tuple<F, RC, CC>(g(arg.template leftCols<dim1>())),
          split_horizontal<F, CCs...>(g(arg.template rightCols<dim2>())));
      }
    }
    else if constexpr (dim1 < column_dimension_of_v<Arg>)
    {
      if constexpr (dynamic_columns<Arg>)
        return detail::make_split_tuple<F, RC, CC>(g(arg.leftCols(dim1)));
      else
        return detail::make_split_tuple<F, RC, CC>(g(arg.template leftCols<dim1>()));
    }
    else
    {
      return detail::make_split_tuple<F, RC, CC>(std::forward<Arg>(arg));
    }
  }*/


  /**
   * \brief Split a matrix horizontally (case in which there is no split).
   */
/*#ifdef __cpp_concepts
  template<typename F = OpenKalman::internal::default_split_function, eigen_matrix Arg>
#else
  template<typename F = OpenKalman::internal::default_split_function, typename Arg,
    std::enable_if_t<eigen_matrix<Arg>, int> = 0>
#endif
  inline auto
  split_horizontal(Arg&& arg) noexcept
  {
    return std::tuple {};
  }*/


  /**
   * \brief Split a matrix horizontally.
   * \tparam CC TypedIndex for the first cut.
   * \tparam CCs TypedIndex for the second and subsequent cuts.
   */
/*#ifdef __cpp_concepts
  template<fixed_index_descriptor CC, fixed_index_descriptor...CCs, eigen_matrix Arg>
  requires (dynamic_columns<Arg> or (dimension_size_of_v<CC> + ... + dimension_size_of_v<CCs>) <= column_dimension_of_v<Arg>)
#else
  template<typename CC, typename...CCs, typename Arg,
    std::enable_if_t<eigen_matrix<Arg> and fixed_index_descriptor<CC> and
      (dynamic_columns<Arg> or (dimension_size_of_v<CC> + ... + dimension_size_of_v<CCs>) <= column_dimension_of<Arg>::value), int> = 0>
#endif
  inline auto
  split_horizontal(Arg&& arg) noexcept
  {
    return split_horizontal<OpenKalman::internal::default_split_function, CC, CCs...>(std::forward<Arg>(arg));
  }*/


  /**
   * \brief Split a matrix horizontally.
   * \tparam cut Number of columns in the first cut.
   * \tparam cuts Numbers of columns in the second and subsequent cuts.
   * \tparam F An object having a static call() method to which the result is applied before creating the tuple.
   */
/*#ifdef __cpp_concepts
  template<std::size_t cut, std::size_t ... cuts, eigen_matrix Arg>
  requires (dynamic_columns<Arg> or (cut + ... + cuts) <= column_dimension_of_v<Arg>)
#else
  template<std::size_t cut, std::size_t ... cuts, typename Arg, std::enable_if_t<eigen_matrix<Arg> and
    (dynamic_columns<Arg> or (cut + ... + cuts) <= column_dimension_of<Arg>::value), int> = 0>
#endif
  inline auto
  split_horizontal(Arg&& arg) noexcept
  {
    return split_horizontal<Dimensions<cut>, Dimensions<cuts>...>(std::forward<Arg>(arg));
  }*/


  /**
   * \brief Split a matrix diagonally and invoke function F on each segment, returning a tuple.
   * \tparam F An object having a static call() method to which the result is applied before creating the tuple.
   * \tparam C TypedIndex for the first cut.
   * \tparam Cs TypedIndex for each of the second and subsequent cuts.
   */
/*#ifdef __cpp_concepts
  template<typename F, bool euclidean, fixed_index_descriptor C, fixed_index_descriptor...Cs, eigen_matrix Arg>
  requires (dynamic_rows<Arg> or
    ((euclidean ? euclidean_dimension_size_of_v<C> : dimension_size_of_v<C>) + ... +
      (euclidean ? euclidean_dimension_size_of_v<Cs> : dimension_size_of_v<Cs>)) <= row_dimension_of_v<Arg>) and
    (dynamic_columns<Arg> or (dimension_size_of_v<C> + ... + dimension_size_of_v<Cs>) <= column_dimension_of_v<Arg>)
#else
  template<typename F, bool euclidean, typename C, typename...Cs, typename Arg, std::enable_if_t<
    eigen_matrix<Arg> and (fixed_index_descriptor<C> and ... and fixed_index_descriptor<Cs>) and
    (dynamic_rows<Arg> or
    ((euclidean ? euclidean_dimension_size_of_v<C> : dimension_size_of_v<C>) + ... +
      (euclidean ? euclidean_dimension_size_of_v<Cs> : dimension_size_of_v<Cs>)) <= row_dimension_of<Arg>::value) and
    (dynamic_columns<Arg> or (dimension_size_of_v<C> + ... + dimension_size_of_v<Cs>) <= column_dimension_of<Arg>::value), int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    // \todo Can g be replaced by make_self_contained?
    constexpr auto g = [](auto&& m) -> decltype(auto) {
      if constexpr (std::is_lvalue_reference_v<Arg&&>)
        return std::forward<decltype(m)>(m);
      else
        return make_dense_writable_matrix_from(std::forward<decltype(m)>(m));
    };

    constexpr Eigen::Index rdim1 = euclidean ? euclidean_dimension_size_of_v<C> : dimension_size_of_v<C>;
    constexpr Eigen::Index cdim1 = dimension_size_of_v<C>;

    if constexpr(sizeof...(Cs) > 0)
    {
      if constexpr (has_dynamic_dimensions<Arg>)
      {
        Eigen::Index rdim2 = get_dimensions_of<0>(arg) - rdim1;
        Eigen::Index cdim2 = get_dimensions_of<1>(arg) - cdim1;

        return std::tuple_cat(
          detail::make_split_tuple<F, C, C>(g(arg.topLeftCorner(rdim1, cdim1))),
          split_diagonal<F, euclidean, Cs...>(g(arg.bottomRightCorner(rdim2, cdim2))));
      }
      else
      {
        constexpr Eigen::Index rdim2 = row_dimension_of_v<Arg> - rdim1;
        constexpr Eigen::Index cdim2 = column_dimension_of_v<Arg> - cdim1;

        return std::tuple_cat(
          detail::make_split_tuple<F, C, C>(g(arg.template topLeftCorner<rdim1, cdim1>())),
          split_diagonal<F, euclidean, Cs...>(g(arg.template bottomRightCorner<rdim2, cdim2>())));
      }
    }
    else if constexpr(rdim1 < std::decay_t<Arg>::RowsAtCompileTime)
    {
      if constexpr (has_dynamic_dimensions<Arg>)
        return detail::make_split_tuple<F, C, C>(g(arg.topLeftCorner(rdim1, cdim1)));
      else
        return detail::make_split_tuple<F, C, C>(g(arg.template topLeftCorner<rdim1, cdim1>()));
    }
    else
    {
      return detail::make_split_tuple<F, C, C>(std::forward<Arg>(arg));
    }
  }*/


  /**
   * \brief Split a matrix diagonally (case in which there is no split).
   */
/*#ifdef __cpp_concepts
  template<typename F = OpenKalman::internal::default_split_function, bool euclidean = false, eigen_matrix Arg>
  requires (not fixed_index_descriptor<F>)
#else
  template<typename F = OpenKalman::internal::default_split_function, bool euclidean = false,
    typename Arg, std::enable_if_t<eigen_matrix<Arg> and not fixed_index_descriptor<F>, int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    return std::tuple {};
  }*/


  /**
   * \brief Split a matrix diagonally by carving out square blocks along the diagonal.
   * \tparam F A class with a static <code>call</code> member to which the result is applied before creating the tuple.
   * \tparam C TypedIndex for the first cut.
   * \tparam Cs TypedIndex for each of the second and subsequent cuts.
   */
/*#ifdef __cpp_concepts
  template<typename F, fixed_index_descriptor C, fixed_index_descriptor...Cs, eigen_matrix Arg>
  requires (not fixed_index_descriptor<F>) and
    (dynamic_rows<Arg> or (dimension_size_of_v<C> + ... + dimension_size_of_v<Cs>) <= row_dimension_of_v<Arg>) and
    (dynamic_columns<Arg> or (dimension_size_of_v<C> + ... + dimension_size_of_v<Cs>) <= column_dimension_of_v<Arg>)
#else
  template<typename F, typename C, typename...Cs, typename Arg, std::enable_if_t<
    eigen_matrix<Arg> and not fixed_index_descriptor<F> and (fixed_index_descriptor<C> and ... and fixed_index_descriptor<Cs>) and
    (dynamic_rows<Arg> or (dimension_size_of_v<C> + ... + dimension_size_of_v<Cs>) <= row_dimension_of<Arg>::value) and
    (dynamic_columns<Arg> or (dimension_size_of_v<C> + ... + dimension_size_of_v<Cs>) <= column_dimension_of<Arg>::value), int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    return split_diagonal<F, false, C, Cs...>(std::forward<Arg>(arg));
  }*/


  /**
   * \brief Split a matrix diagonally by carving out square blocks along the diagonal.
   * \tparam C TypedIndex for the first cut.
   * \tparam Cs TypedIndex for the second and subsequent cuts.
   */
/*#ifdef __cpp_concepts
  template<fixed_index_descriptor C, fixed_index_descriptor...Cs, eigen_matrix Arg>
  requires (dynamic_rows<Arg> or (dimension_size_of_v<C> + ... + dimension_size_of_v<Cs>) <= row_dimension_of_v<Arg>) and
    (dynamic_columns<Arg> or (dimension_size_of_v<C> + ... + dimension_size_of_v<Cs>) <= column_dimension_of_v<Arg>)
#else
  template<typename C, typename...Cs, typename Arg, std::enable_if_t<
    eigen_matrix<Arg> and (fixed_index_descriptor<C> and ... and fixed_index_descriptor<Cs>) and
    (dynamic_rows<Arg> or (dimension_size_of_v<C> + ... + dimension_size_of_v<Cs>) <= row_dimension_of<Arg>::value) and
    (dynamic_columns<Arg> or (dimension_size_of_v<C> + ... + dimension_size_of_v<Cs>) <= column_dimension_of<Arg>::value), int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    return split_diagonal<OpenKalman::internal::default_split_function, C, Cs...>(std::forward<Arg>(arg));
  }*/


  /**
   * \brief Split a matrix diagonally by carving out square blocks along the diagonal.
   * \tparam cut Number of rows and columns in the first cut.
   * \tparam cuts Numbers of rows and columns in the second and subsequent cuts.
   * \tparam F An object having a static call() method to which the result is applied before creating the tuple.
   */
/*#ifdef __cpp_concepts
  template<std::size_t cut, std::size_t ... cuts, eigen_matrix Arg>
  requires (dynamic_columns<Arg> or (cut + ... + cuts) <= column_dimension_of_v<Arg>) and
    (dynamic_rows<Arg> or (cut + ... + cuts) <= row_dimension_of_v<Arg>)
#else
  template<std::size_t cut, std::size_t ... cuts, typename Arg,
    std::enable_if_t<eigen_matrix<Arg> and
    (dynamic_columns<Arg> or (cut + ... + cuts) <= column_dimension_of<Arg>::value) and
    (dynamic_rows<Arg> or (cut + ... + cuts) <= row_dimension_of<Arg>::value), int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    return split_diagonal<Dimensions<cut>, Dimensions<cuts>...>(std::forward<Arg>(arg));
  }*/


  // \todo Add functions that return stl-compatible iterators.

  // ======================= //
  //  Block-wise operations  //
  // ======================= //


  namespace detail
  {
#ifdef __cpp_concepts
    template<typename Function, typename Arg, typename...is>
    concept col_result_is_lvalue = writable<Arg> and std::is_lvalue_reference_v<Arg> and
      requires(const Function& f, std::decay_t<decltype(column(std::declval<Arg&&>(), 0))>& col, is...i) {
        {col} -> writable;
        requires requires { {f(col, i...)} -> std::same_as<void>; } or
          requires { {f(col, i...)} -> std::same_as<decltype(col)>; };
      };
#else
    template<typename Function, typename Arg, bool isvoid, typename = void, typename...is>
    struct col_result_is_valid : std::false_type {};

    template<typename Function, typename Arg, bool isvoid, typename...is>
    struct col_result_is_valid<Function, Arg, isvoid, std::enable_if_t<
      writable<decltype(column(std::declval<Arg&&>(), 0))> and std::is_same_v<
        typename std::invoke_result<Function, std::decay_t<decltype(column(std::declval<Arg&&>(), 0))>&, is...>::type,
        std::conditional_t<isvoid, void, std::decay_t<decltype(column(std::declval<Arg&&>(), 0))>&>>>, is...>
      : std::true_type {};

    template<typename Function, typename Arg, typename...is>
    static constexpr bool col_result_is_lvalue = writable<Arg> and std::is_lvalue_reference_v<Arg> and
      (col_result_is_valid<Function, Arg, true, void, is...>::value or
        col_result_is_valid<Function, Arg, false, void, is...>::value);


    template<typename Function, typename Arg, typename = void, typename...is>
    struct col_result_is_column_impl : std::false_type {};

    template<typename Function, typename Arg, typename...is>
    struct col_result_is_column_impl<Function, Arg, std::enable_if_t<
      column_vector<typename std::invoke_result<Function, decltype(column(std::declval<Arg&&>(), 0)), is...>::type>
      >, is...> : std::true_type {};

    template<typename Function, typename Arg, typename...is>
    static constexpr bool col_result_is_column_vector = col_result_is_column_impl<Function, Arg, void, is...>::value;
#endif


    template<bool index, typename F, typename Arg, std::size_t ... ints>
    inline decltype(auto) columnwise_impl(const F& f, Arg&& arg, std::index_sequence<ints...>)
    {
      static_assert(not dynamic_columns<Arg>);

      if constexpr ((index and detail::col_result_is_lvalue<F, Arg, std::size_t>) or
        (not index and detail::col_result_is_lvalue<F, Arg>))
      {
        static_assert(std::is_lvalue_reference_v<Arg>);

        return ([](const F& f, Arg&& arg) -> Arg& {
          decltype(auto) c {column<ints>(arg)};
          static_assert(writable<decltype(c)>);
          if constexpr (index) f(c, ints); else f(c);
          return arg;
        }(f, arg), ...);
      }
      else
      {
        if constexpr (index)
          return concatenate_horizontal(f(column<ints>(std::forward<Arg>(arg)), ints)...);
        else
          return concatenate_horizontal(f(column<ints>(std::forward<Arg>(arg)))...);
      }
    };


  } // namespace detail


/*#ifdef __cpp_concepts
  template<typename Function, eigen_matrix Arg> requires
    detail::col_result_is_lvalue<Function, Arg> or detail::col_result_is_lvalue<Function, Arg, std::size_t> or
    requires(const Function& f, Arg&& arg) {{f(column(std::forward<Arg>(arg), 0))} -> column_vector; } or
    requires(const Function& f, Arg&& arg, std::size_t i) {{f(column(std::forward<Arg>(arg), 0), i)} -> column_vector; }
#else
  template<typename Function, typename Arg, std::enable_if_t<eigen_matrix<Arg> and
    (detail::col_result_is_lvalue<Function, Arg> or detail::col_result_is_lvalue<Function, Arg, std::size_t> or
    detail::col_result_is_column_vector<Function, Arg> or
    detail::col_result_is_column_vector<Function, Arg, std::size_t>), int> = 0>
#endif
  inline decltype(auto)
  apply_columnwise(const Function& f, Arg&& arg)
  {
    constexpr bool index = detail::col_result_is_lvalue<Function, Arg, std::size_t> or
#ifdef __cpp_concepts
      requires(const Function& f, Arg&& arg, std::size_t i) {
        {f(column(std::forward<Arg>(arg), 0), i)} -> column_vector;};
#else
      detail::col_result_is_column_vector<Function, Arg, std::size_t>;
#endif

    if constexpr (dynamic_columns<Arg>)
    {
      auto cols = get_dimensions_of<1>(arg);

      if constexpr ((index and detail::col_result_is_lvalue<Function, Arg, std::size_t>) or
        (not index and detail::col_result_is_lvalue<Function, Arg>))
      {
        static_assert(std::is_lvalue_reference_v<Arg>);

        for (std::size_t j = 0; j<cols; j++)
        {
          decltype(auto) c {column(arg, j)};
          static_assert(writable<decltype(c)>);
          if constexpr (index) f(c, j); else f(c);
        }
        return (arg);
      }
      else
      {
        auto res_col0 = [](const Function& f, Arg&& arg){
          auto col0 = column(std::forward<Arg>(arg), 0);
          if constexpr (index) return f(col0, 0); else return f(col0);
        }(f, std::forward<Arg>(arg));

        using ResultType = decltype(res_col0);
        using M = eigen_matrix_t<scalar_type_of_t<ResultType>, row_dimension_of_v<ResultType>, dynamic_size>;
        M m {get_dimensions_of<0>(res_col0), cols};

        column(m, 0) = res_col0;

        for (std::size_t j = 1; j<cols; j++)
        {
          if constexpr (index)
            column(m, j) = f(column(std::forward<Arg>(arg), j), j);
          else
            column(m, j) = f(column(std::forward<Arg>(arg), j));
        }
        return m;
      }
    }
    else
    {
      return detail::columnwise_impl<index>(
        f, std::forward<Arg>(arg), std::make_index_sequence<column_dimension_of_v<Arg>> ());
    }
  }*/


  /*namespace detail
  {
    template<typename Function, std::size_t ... ints>
    inline auto cat_columnwise_impl(const Function& f, std::index_sequence<ints...>)
    {
      return concatenate_horizontal(f(ints)...);
    };

#ifndef __cpp_concepts
    template<typename Function, typename = void, typename...is>
    struct columns_1_or_dynamic_impl : std::false_type {};

    template<typename Function, typename...is>
    struct columns_1_or_dynamic_impl<Function, std::enable_if_t<
      eigen_matrix<std::invoke_result_t<const Function&, is...>> and
      ( dynamic_columns<std::invoke_result_t<const Function&, is...>> or
        column_vector<std::invoke_result_t<const Function&, is...>>)>, is...> : std::true_type {};

    template<typename Function, typename...is>
    static constexpr bool columns_1_or_dynamic = columns_1_or_dynamic_impl<Function, void, is...>::value;
#endif
  }*/


/*#ifdef __cpp_concepts
  template<std::size_t...compile_time_columns, typename Function, std::convertible_to<std::size_t>...runtime_columns>
  requires (sizeof...(compile_time_columns) + sizeof...(runtime_columns) == 1) and
    ( requires(const Function& f) {
      {f()} -> eigen_matrix;
      requires requires { {f()} -> dynamic_columns; } or requires { {f()} -> column_vector; };
    } or requires(const Function& f, std::size_t& i) {
      {f(i)} -> eigen_matrix;
      requires requires { {f(i)} -> dynamic_columns; } or requires { {f(i)} -> column_vector; };
    })
#else
  template<std::size_t...compile_time_columns, typename Function, typename...runtime_columns, std::enable_if_t<
    (std::is_convertible_v<runtime_columns, std::size_t> and ...) and
    (sizeof...(compile_time_columns) + sizeof...(runtime_columns) == 1), int> = 0>
#endif
  inline auto
  apply_columnwise(const Function& f, runtime_columns...c)
  {
#ifdef __cpp_concepts
    if constexpr (requires(std::size_t& i) {
      {f(i)} -> eigen_matrix;
      requires requires { {f(i)} -> dynamic_columns; } or requires { {f(i)} -> column_vector; }; })
#else
    static_assert(detail::columns_1_or_dynamic<Function> or detail::columns_1_or_dynamic<Function, std::size_t&>);

    if constexpr (detail::columns_1_or_dynamic<Function, std::size_t&>)
#endif
    {
      if constexpr (sizeof...(compile_time_columns) > 0)
      {
        return detail::cat_columnwise_impl(f, std::make_index_sequence<(compile_time_columns + ... + 0)> {});
      }
      else
      {
        using R = std::invoke_result_t<const Function&, std::size_t>;
        using Scalar = scalar_type_of_t<R>;
        std::size_t cols = (c + ... + 0);

        if constexpr (dynamic_rows<R>)
        {
          auto r0 = f(0);
          auto rows = get_dimensions_of<0>(r0);
          eigen_matrix_t<Scalar, dynamic_size, dynamic_size> m {rows, cols};
          m.col(0) = r0;
          for (std::size_t j = 1; j < cols; j++)
          {
            auto rj = f(j);
            assert(get_dimensions_of<0>(rj) == rows);
            m.col(j) = rj;
          }
          return m;
        }
        else
        {
          constexpr auto rows = row_dimension_of_v<R>;
          eigen_matrix_t<Scalar, rows, dynamic_size> m {rows, cols};
          for (std::size_t j = 0; j < cols; j++)
          {
            m.col(j) = f(j);
          }
          return m;
        }
      }
    }
    else
    {
      auto r = f();
      using R = decltype(r);
      if constexpr (dynamic_columns<R>) assert (get_dimensions_of<1>(r) == 1);

      if constexpr (sizeof...(compile_time_columns) > 0)
        return make_self_contained(Eigen::Replicate<R, 1, compile_time_columns...>(r));
      else
        return make_self_contained(Eigen::Replicate<R, 1, Eigen::Dynamic>(r, 1, c...));
    }
  }*/


  /*namespace detail
  {
#ifdef __cpp_concepts
    template<typename Function, typename Arg, typename...is>
    concept row_result_is_lvalue = (not std::is_const_v<Arg>) and std::is_lvalue_reference_v<Arg> and
      requires(const Function& f, std::decay_t<decltype(row(std::declval<Arg&&>(), 0))>& row, is...i) {
        {row} -> writable;
        requires requires { {f(row, i...)} -> std::same_as<void>; } or
          requires { {f(row, i...)} -> std::same_as<decltype(row)>; };
      };
#else
    template<typename Function, typename Arg, bool isvoid, typename = void, typename...is>
    struct row_result_is_valid : std::false_type {};

    template<typename Function, typename Arg, bool isvoid, typename...is>
    struct row_result_is_valid<Function, Arg, isvoid, std::enable_if_t<
      writable<decltype(row(std::declval<Arg&&>(), 0))> and std::is_same_v<
        typename std::invoke_result<Function, std::decay_t<decltype(row(std::declval<Arg&&>(), 0))>&, is...>::type,
        std::conditional_t<isvoid, void, std::decay_t<decltype(row(std::declval<Arg&&>(), 0))>&>>>, is...>
      : std::true_type {};

    template<typename Function, typename Arg, typename...is>
    static constexpr bool row_result_is_lvalue = writable<Arg> and std::is_lvalue_reference_v<Arg> and
      (row_result_is_valid<Function, Arg, true, void, is...>::value or
        row_result_is_valid<Function, Arg, false, void, is...>::value);


    template<typename Function, typename Arg, typename = void, typename...is>
    struct row_result_is_row_impl : std::false_type {};

    template<typename Function, typename Arg, typename...is>
    struct row_result_is_row_impl<Function, Arg, std::enable_if_t<
      row_vector<typename std::invoke_result<Function, decltype(row(std::declval<Arg&&>(), 0)), is...>::type>
      >, is...> : std::true_type {};

    template<typename Function, typename Arg, typename...is>
    static constexpr bool row_result_is_row_vector = row_result_is_row_impl<Function, Arg, void, is...>::value;
#endif*/


    /*template<bool index, typename F, typename Arg, std::size_t ... ints>
    inline decltype(auto) rowwise_impl(const F& f, Arg&& arg, std::index_sequence<ints...>)
    {
      if constexpr ((index and detail::row_result_is_lvalue<F, Arg, std::size_t>) or
        (not index and detail::row_result_is_lvalue<F, Arg>))
      {
        static_assert(std::is_lvalue_reference_v<Arg>);

        return ([](const F& f, Arg&& arg) -> Arg& {
          decltype(auto) r {row<ints>(arg)};
          static_assert(writable<decltype(r)>);
          if constexpr (index) f(r, ints); else f(r);
          return arg;
        }(f, arg), ...);
      }
      else
      {
        if constexpr (index)
          return concatenate_vertical(f(row<ints>(std::forward<Arg>(arg)), ints)...);
        else
          return concatenate_vertical(f(row<ints>(std::forward<Arg>(arg)))...);
      }
    };

  }*/ // namespace detail


/*#ifdef __cpp_concepts
  template<typename Function, eigen_matrix Arg> requires
    detail::row_result_is_lvalue<Function, Arg> or detail::row_result_is_lvalue<Function, Arg, std::size_t> or
    requires(const Function& f, Arg&& arg) {{f(row(std::forward<Arg>(arg), 0))} -> row_vector; } or
    requires(const Function& f, Arg&& arg, std::size_t i) {{f(row(std::forward<Arg>(arg), 0), i)} -> row_vector; }
#else
  template<typename Function, typename Arg, std::enable_if_t<eigen_matrix<Arg> and
    (detail::row_result_is_lvalue<Function, Arg> or detail::row_result_is_lvalue<Function, Arg, std::size_t> or
    detail::row_result_is_row_vector<Function, Arg> or
    detail::row_result_is_row_vector<Function, Arg, std::size_t>), int> = 0>
#endif
  inline decltype(auto)
  apply_rowwise(const Function& f, Arg&& arg)
  {
    constexpr bool index = detail::row_result_is_lvalue<Function, Arg, std::size_t> or
#ifdef __cpp_concepts
      requires(const Function& f, Arg&& arg, std::size_t i) {{f(row(std::forward<Arg>(arg), 0), i)} -> row_vector;};
#else
      detail::row_result_is_row_vector<Function, Arg, std::size_t>;
#endif

    if constexpr (dynamic_rows<Arg>)
    {
      auto rows = get_dimensions_of<0>(arg);

      if constexpr ((index and detail::row_result_is_lvalue<Function, Arg, std::size_t>) or
        (not index and detail::row_result_is_lvalue<Function, Arg>))
      {
        static_assert(std::is_lvalue_reference_v<Arg>);

        for (std::size_t i = 0; i<rows; i++)
        {
          decltype(auto) r {row(arg, i)};
          static_assert(writable<decltype(r)>);
          if constexpr (index) f(r, i); else f(r);
        }
        return (arg);
      }
      else
      {
        auto res_row0 = [](const Function& f, Arg&& arg){
          auto row0 = row(std::forward<Arg>(arg), 0);
          if constexpr (index) return f(row0, 0); else return f(row0);
        }(f, std::forward<Arg>(arg));

        using ResultType = decltype(res_row0);
        using M = eigen_matrix_t<
          scalar_type_of_t<ResultType>, dynamic_size, column_dimension_of_v<ResultType>>;
        M m {rows, get_dimensions_of<1>(res_row0)};

        row(m, 0) = res_row0;

        for (std::size_t i = 1; i<rows; i++)
        {
          if constexpr (index)
            row(m, i) = f(row(std::forward<Arg>(arg), i), i);
          else
            row(m, i) = f(row(std::forward<Arg>(arg), i));
        }
        return m;
      }
    }
    else
    {
      return detail::rowwise_impl<index>(f, std::forward<Arg>(arg), std::make_index_sequence<row_dimension_of_v<Arg>> {});
    }
  }*/


  /*namespace detail
  {
    template<typename Function, std::size_t ... ints>
    inline auto cat_rowwise_impl(const Function& f, std::index_sequence<ints...>)
    {
      return concatenate_vertical(f(ints)...);
    };

#ifndef __cpp_concepts
    template<typename Function, typename = void, typename...is>
    struct rows_1_or_dynamic_impl : std::false_type {};

    template<typename Function, typename...is>
    struct rows_1_or_dynamic_impl<Function, std::enable_if_t<
      eigen_matrix<std::invoke_result_t<const Function&, is...>> and
      ( dynamic_rows<std::invoke_result_t<const Function&, is...>> or
        row_vector<std::invoke_result_t<const Function&, is...>>)>, is...> : std::true_type {};

    template<typename Function, typename...is>
    static constexpr bool rows_1_or_dynamic = rows_1_or_dynamic_impl<Function, void, is...>::value;
#endif
  }*/


/*#ifdef __cpp_concepts
  template<std::size_t...compile_time_rows, typename Function, std::convertible_to<std::size_t>...runtime_rows> requires
    (sizeof...(compile_time_rows) + sizeof...(runtime_rows) == 1) and
    ( requires(const Function& f) {
      {f()} -> eigen_matrix;
      requires requires { {f()} -> dynamic_rows; } or requires { {f()} -> row_vector; };
    } or requires(const Function& f, std::size_t& i) {
      {f(i)} -> eigen_matrix;
      requires requires { {f(i)} -> dynamic_rows; } or requires { {f(i)} -> row_vector; };
    })
#else
  template<std::size_t...compile_time_rows, typename Function, typename...runtime_rows, std::enable_if_t<
    (std::is_convertible_v<runtime_rows, std::size_t> and ...) and
    (sizeof...(compile_time_rows) + sizeof...(runtime_rows) == 1), int> = 0>
#endif
  inline auto
  apply_rowwise(const Function& f, runtime_rows...r)
  {
#ifdef __cpp_concepts
    if constexpr (requires(std::size_t& i) {
      {f(i)} -> eigen_matrix;
      requires requires { {f(i)} -> dynamic_rows; } or requires { {f(i)} -> row_vector; }; })
#else
    static_assert(detail::rows_1_or_dynamic<Function> or detail::rows_1_or_dynamic<Function, std::size_t&>);

    if constexpr (detail::rows_1_or_dynamic<Function, std::size_t&>)
#endif
    {
      if constexpr (sizeof...(runtime_rows) == 0)
      {
        return detail::cat_rowwise_impl(f, std::make_index_sequence<(compile_time_rows + ... + 0)> {});
      }
      else
      {
        using R = std::invoke_result_t<const Function&, std::size_t>;
        using Scalar = scalar_type_of_t<R>;
        std::size_t rows = (r + ... + 0);

        if constexpr (dynamic_columns<R>)
        {
          auto c0 = f(0);
          auto cols = get_dimensions_of<1>(c0);
          eigen_matrix_t<Scalar, dynamic_size, dynamic_size> m {rows, cols};
          m.row(0) = c0;
          for (std::size_t i = 1; i < rows; i++)
          {
            auto ci = f(i);
            assert(get_dimensions_of<1>(ci) == cols);
            m.row(i) = ci;
          }
          return m;
        }
        else
        {
          constexpr auto cols = column_dimension_of_v<R>;
          eigen_matrix_t<Scalar, dynamic_size, cols> m {rows, cols};
          for (std::size_t i = 0; i < rows; i++)
          {
            m.row(i) = f(i);
          }
          return m;
        }
      }
    }
    else
    {
      auto c = f();
      using C = decltype(c);
      if constexpr (dynamic_rows<C>) assert (get_dimensions_of<0>(c) == 1);

      if constexpr (sizeof...(compile_time_rows) > 0)
        return make_self_contained(Eigen::Replicate<C, compile_time_rows..., 1>(c));
      else
        return make_self_contained(Eigen::Replicate<C, Eigen::Dynamic, 1>(c, r..., 1));
    }
  }*/


} // namespace OpenKalman

#endif //OPENKALMAN_SUBSET_FUNCTIONS_HPP
