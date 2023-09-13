/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_FIXEDSIZEADAPTER_HPP
#define OPENKALMAN_FIXEDSIZEADAPTER_HPP

namespace OpenKalman::internal
{
#ifdef __cpp_concepts
  template<indexible NestedMatrix, index_descriptor...IndexDescriptors>
    requires compatible_with_index_descriptors<NestedMatrix, IndexDescriptors...> and
      (sizeof...(IndexDescriptors) == max_indices_of_v<NestedMatrix> or
        (sizeof...(IndexDescriptors) == 0 and one_by_one_matrix<NestedMatrix, Likelihood::maybe>))
#else
  template<typename NestedMatrix, typename...IndexDescriptors>
#endif
  struct FixedSizeAdapter : library_base<FixedSizeAdapter<NestedMatrix, IndexDescriptors...>, NestedMatrix>
  {

#ifndef __cpp_concepts
    static_assert(indexible<NestedMatrix>);
    static_assert((index_descriptor<IndexDescriptors> and ...));
    static_assert(compatible_with_index_descriptors<NestedMatrix, IndexDescriptors...>);
    static_assert(sizeof...(IndexDescriptors) == max_indices_of_v<NestedMatrix> or
      (sizeof...(IndexDescriptors) == 0 and one_by_one_matrix<NestedMatrix, Likelihood::maybe>));
#endif


    /**
     * \brief Construct from compatible indexible object.
     */
#ifdef __cpp_concepts
    template<compatible_with_index_descriptors<IndexDescriptors...> Arg>
      requires (not std::derived_from<std::decay_t<Arg>, FixedSizeAdapter>) and std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<(not std::is_base_of_v<FixedSizeAdapter, std::decay_t<Arg>>) and
      compatible_with_index_descriptors<Arg, IndexDescriptors...> and std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    explicit FixedSizeAdapter(Arg&& arg) noexcept : m_arg {std::forward<Arg>(arg)} {}


    /**
     * \brief Construct from compatible indexible object based on a set of fixed index descriptors.
     */
#ifdef __cpp_concepts
    template<compatible_with_index_descriptors<IndexDescriptors...> Arg, fixed_index_descriptor...Ids>
      requires (... and (dynamic_index_descriptor<IndexDescriptors> or equivalent_to<Ids, IndexDescriptors>)) and
      std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, typename...Ids, std::enable_if_t<
      compatible_with_index_descriptors<Arg, IndexDescriptors...> and (... and fixed_index_descriptor<Ids>) and
      (... and (dynamic_index_descriptor<IndexDescriptors> or equivalent_to<Ids, IndexDescriptors>)) and
      std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    FixedSizeAdapter(Arg&& arg, const Ids&...) noexcept : m_arg {std::forward<Arg>(arg)} {}


    /**
     * \overload
     * \brief Construct from another FixedSizeAdapter using a set of fixed index descriptors.
     */
#ifdef __cpp_concepts
    template<fixed_size_adapter Arg, fixed_index_descriptor...Ids> requires
        compatible_with_index_descriptors<nested_matrix_of_t<Arg>, IndexDescriptors...> and
        (... and (dynamic_index_descriptor<IndexDescriptors> or equivalent_to<Ids, IndexDescriptors>)) and
      std::constructible_from<NestedMatrix, nested_matrix_of_t<Arg&&>>
#else
    template<typename Arg, typename...Ids, std::enable_if_t<fixed_size_adapter<Arg> and
      compatible_with_index_descriptors<nested_matrix_of_t<Arg>, IndexDescriptors...> and (... and fixed_index_descriptor<Ids>) and
      (... and (dynamic_index_descriptor<IndexDescriptors> or equivalent_to<Ids, IndexDescriptors>)) and
      std::is_constructible_v<NestedMatrix, nested_matrix_of_t<Arg&&>>, int> = 0>
#endif
    FixedSizeAdapter(Arg&& arg, const Ids&...) noexcept : m_arg {std::forward<Arg>(arg).nested_matrix()} {}


    /**
     * \brief Assign from another compatible indexible object.
     */
#ifdef __cpp_concepts
    template<compatible_with_index_descriptors<IndexDescriptors...> Arg> requires
    std::assignable_from<std::add_lvalue_reference_t<NestedMatrix>, decltype(to_native_matrix<NestedMatrix>(std::declval<Arg&&>()))>
#else
    template<typename Arg, std::enable_if_t<compatible_with_index_descriptors<Arg, IndexDescriptors...> and
      std::is_assignable_v<std::add_lvalue_reference_t<NestedMatrix>, decltype(to_native_matrix<NestedMatrix>(std::declval<Arg&&>()))>, int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      m_arg = to_native_matrix<NestedMatrix>(std::forward<Arg>(arg));
      return *this;
    }


    /**
     * \brief Get the nested matrix.
     */
    const NestedMatrix& nested_matrix() & noexcept { return m_arg; }

    /// \overload
    const NestedMatrix& nested_matrix() const & noexcept { return m_arg; }

    /// \overload
    NestedMatrix nested_matrix() && noexcept { return std::move(m_arg); }

    /// \overload
    NestedMatrix nested_matrix() const && noexcept { return std::move(m_arg); }


    /**
     * \brief Increment from another indexible object.
     */
#ifdef __cpp_concepts
    template<compatible_with_index_descriptors<IndexDescriptors...> Arg>
#else
    template<typename Arg, std::enable_if_t<compatible_with_index_descriptors<Arg, IndexDescriptors...>, int> = 0>
#endif
    auto& operator+=(Arg&& arg) noexcept
    {
      this->nested_matrix() += to_native_matrix<NestedMatrix>(std::forward<Arg>(arg));
      return *this;
    }


    /**
     * \brief Decrement from another indexible object.
     */
#ifdef __cpp_concepts
    template<compatible_with_index_descriptors<IndexDescriptors...> Arg>
#else
    template<typename Arg, std::enable_if_t<compatible_with_index_descriptors<Arg, IndexDescriptors...>, int> = 0>
#endif
    auto& operator-=(Arg&& arg) noexcept
    {
      this->nested_matrix() -= to_native_matrix<NestedMatrix>(std::forward<Arg>(arg));
      return *this;
    }

  private:

    NestedMatrix m_arg; //< The nested matrix.

  };


  // ----------------- //
  //  Deduction Guide  //
  // ----------------- //

#ifdef __cpp_concepts
    template<indexible Arg, fixed_index_descriptor...Ids> requires (not fixed_size_adapter<Arg>) and (sizeof...(Ids) > 0)
#else
    template<typename Arg, typename...Ids, std::enable_if_t<indexible<Arg> and not fixed_size_adapter<Arg> and
      (... and fixed_index_descriptor<Ids>) and (sizeof...(Ids) > 0), int> = 0>
#endif
    FixedSizeAdapter(Arg&&, const Ids&...) -> FixedSizeAdapter<std::remove_reference_t<Arg>, Ids...>;


#ifdef __cpp_concepts
    template<fixed_size_adapter Arg, fixed_index_descriptor...Ids> requires (sizeof...(Ids) > 0)
#else
    template<typename Arg, typename...Ids, std::enable_if_t<fixed_size_adapter<Arg> and
      (... and fixed_index_descriptor<Ids>) and (sizeof...(Ids) > 0), int> = 0>
#endif
    FixedSizeAdapter(Arg&&, const Ids&...) -> FixedSizeAdapter<std::decay_t<nested_matrix_of_t<Arg>>, Ids...>;


#ifdef __cpp_concepts
    template<one_by_one_matrix<Likelihood::maybe> Arg>
#else
    template<typename Arg, std::enable_if_t<one_by_one_matrix<Arg, Likelihood::maybe>, int> = 0>
#endif
    FixedSizeAdapter(Arg&&) -> FixedSizeAdapter<std::remove_reference_t<Arg>>;


} // namespace OpenKalman::internal


// ------------------------- //
//   IndexibleObjectTraits   //
// ------------------------- //

namespace OpenKalman::interface
{
  template<typename NestedMatrix, typename...IndexDescriptors>
  struct IndexibleObjectTraits<internal::FixedSizeAdapter<NestedMatrix, IndexDescriptors...>>
  {
    static constexpr std::size_t max_indices = sizeof...(IndexDescriptors);

    using index_type = index_type_of_t<NestedMatrix>;

    using scalar_type = scalar_type_of_t<NestedMatrix>;

    template<typename Arg, typename N>
    static constexpr auto get_index_descriptor(const Arg& arg, N n)
    {
      if constexpr (static_index_value<N>)
      {
        using ID = std::tuple_element_t<static_index_value_of_v<N>, std::tuple<IndexDescriptors...>>;
        if constexpr (fixed_index_descriptor<ID>) return ID {};
        else return OpenKalman::get_index_descriptor(arg.nested_matrix(), n);
      }
      else if constexpr (equivalent_to<IndexDescriptors...>)
      {
        using ID = std::tuple_element_t<0, std::tuple<IndexDescriptors...>>;
        if constexpr (fixed_index_descriptor<ID>) return ID {};
        else return OpenKalman::get_index_descriptor<N>(arg.nested_matrix(), n);
      }
      else
      {
        return OpenKalman::get_index_descriptor<N>(arg.nested_matrix(), n);
      }
    }

    static constexpr bool has_runtime_parameters = false;

    using type = std::tuple<NestedMatrix>;

    template<std::size_t i, typename Arg>
    static decltype(auto) get_nested_matrix(Arg&& arg)
    {
      static_assert(i == 0);
      return std::forward<Arg>(arg).nested_matrix();
    }

    template<typename Arg>
    static auto convert_to_self_contained(Arg&& arg)
    {
      return make_self_contained(std::forward<Arg>(arg).nested_matrix());
    }

  private:

    static constexpr bool maybe_1x1 =
      ((dynamic_index_descriptor<IndexDescriptors> or dimension_size_of_v<IndexDescriptors> == 1) and ...);

  public:

    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (constant_matrix<NestedMatrix, CompileTimeStatus::any, Likelihood::maybe> and
          (constant_matrix<NestedMatrix> or maybe_1x1))
        return constant_coefficient{arg.nested_matrix()};
      else
        return std::monostate {};
    }

    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
      if constexpr (constant_diagonal_matrix<NestedMatrix, CompileTimeStatus::any, Likelihood::maybe> and
          (constant_diagonal_matrix<NestedMatrix> or maybe_1x1))
        return constant_diagonal_coefficient {arg.nested_matrix()};
      else
        return std::monostate {};
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = triangular_matrix<NestedMatrix, t, b>;

    static constexpr bool is_triangular_adapter = false;

    template<Likelihood b>
    static constexpr bool is_diagonal_adapter = false;

    template<TriangleType t, typename Arg>
    static constexpr auto make_triangular_matrix(Arg&& arg)
    {
      return make_triangular_matrix<NestedMatrix>(std::forward<Arg>(arg));
    }

    static constexpr bool is_hermitian = hermitian_matrix<NestedMatrix, Likelihood::maybe>;

    template<HermitianAdapterType t, typename Arg>
    static constexpr auto make_hermitian_adapter(Arg&& arg)
    {
      return make_hermitian_matrix<t>(std::forward<Arg>(arg).nested_matrix());
    }

#ifdef __cpp_lib_concepts
    template<typename Arg, typename...I> requires element_gettable<decltype(std::declval<Arg&&>().nested_matrix()), sizeof...(I)>
#else
    template<typename Arg, typename...I, std::enable_if_t<element_gettable<decltype(std::declval<Arg&&>().nested_matrix()), sizeof...(I)>, int> = 0>
#endif
    static constexpr decltype(auto) get(Arg&& arg, I...i)
    {
      return get_element(std::forward<Arg>(arg).nested_matrix(), i...);
    }


#ifdef __cpp_lib_concepts
    template<typename Arg, typename...I> requires element_settable<decltype(std::declval<Arg&>().nested_matrix()), sizeof...(I)>
#else
    template<typename Arg, typename...I, std::enable_if_t<element_settable<decltype(std::declval<Arg&>().nested_matrix()), sizeof...(I)>, int> = 0>
#endif
    static constexpr void set(Arg& arg, const scalar_type_of_t<Arg>& s, I...i)
    {
      arg.set_element(arg.nested_matrix(), s, i...);
    }


    static constexpr bool is_writable = writable<NestedMatrix>;


#ifdef __cpp_lib_concepts
    template<typename Arg> requires directly_accessible<nested_matrix_of_t<Arg&>>
#else
    template<typename Arg, std::enable_if_t<directly_accessible<typename nested_matrix_of<Arg&>::type>, int> = 0>
#endif
    static constexpr auto*
    data(Arg& arg) { return internal::raw_data(arg.nested_matrix()); }


    static constexpr Layout layout = layout_of_v<NestedMatrix>;

  };


  // ----------------- //
  //  LibraryRoutines  //
  // ----------------- //

  template<typename NestedMatrix, typename...IndexDescriptors>
  struct LibraryRoutines<internal::FixedSizeAdapter<NestedMatrix, IndexDescriptors...>>
  {
  private:

    using Nested = std::decay_t<NestedMatrix>;
    using NestedInterface = LibraryRoutines<Nested>;

  public:

    template<typename Derived>
    using LibraryBase = internal::library_base<Derived, NestedMatrix>;


    template<typename Arg>
    static decltype(auto) to_native_matrix(Arg&& arg)
    {
      return OpenKalman::to_native_matrix<Nested>(std::forward<Arg>(arg));
    }


    template<typename C, typename...D>
    static constexpr auto make_constant_matrix(C&& c, D&&...d)
    {
      return make_constant_matrix_like<Nested>(std::forward<C>(c), std::forward<D>(d)...);
    }


    template<typename Scalar, typename D>
    static constexpr auto make_identity_matrix(D&& d)
    {
      return make_identity_matrix_like<Nested, Scalar>(std::forward<D>(d));
    }


    template<typename Arg, typename...Begin, typename...Size>
    static decltype(auto) get_block(Arg&& arg, std::tuple<Begin...> begin, std::tuple<Size...> size)
    {
      return OpenKalman::get_block(std::forward<Arg>(arg).nested_matrix(), begin, size);
    };


    template<typename Arg, typename Block, typename...Begin>
    static Arg& set_block(Arg& arg, Block&& block, Begin...begin)
    {
      OpenKalman::set_block(std::forward<Arg>(arg).nested_matrix(), to_native_matrix(std::forward<Block>(block)), begin...);
      return arg;
    };


    template<TriangleType t, typename A, typename B>
    static auto set_triangle(A&& a, B&& b)
    {
      return internal::FixedSizeAdapter<Nested, IndexDescriptors...> {
        OpenKalman::internal::set_triangle<t>(std::forward<A>(a).nested_matrix(), to_native_matrix(std::forward<B>(b)))};
    }


    template<typename Arg>
    static constexpr decltype(auto)
    to_diagonal(Arg&& arg)
    {
      if constexpr (one_by_one_matrix<Arg>) return std::forward<Arg>(arg);
      else return OpenKalman::to_diagonal(std::forward<Arg>(arg).nested_matrix());
    }


    template<typename Arg>
    static constexpr decltype(auto)
    diagonal_of(Arg&& arg)
    {
      decltype(auto) ret = OpenKalman::diagonal_of(std::forward<Arg>(arg).nested_matrix());
      using Ret = decltype(ret);
      using D = std::decay_t<decltype(*get_is_square(ret))>;
      if constexpr (dynamic_dimension<Ret, 0> and not dynamic_index_descriptor<D>)
        return internal::FixedSizeAdapter<NestedMatrix, D> {std::forward<Ret>(ret)};
      else
        return std::forward<Ret>(ret);
    }

  private:

    template<typename Arg, std::size_t...Ix, typename...Factors>
    static constexpr auto broadcast_impl(Arg&& arg, std::index_sequence<Ix...>, const Factors&...factors)
    {
      if constexpr (((dynamic_dimension<Arg, Ix> and not dynamic_index_descriptor<IndexDescriptors> and static_index_value<Factors>) or ...))
        return internal::FixedSizeAdapter {std::forward<Arg>(arg),
          replicate_index_descriptor(get_index_descriptor<Ix>(arg), factors)...};
      else
        return std::forward<Arg>(arg);
    }

  public:

    template<typename Arg, typename...Factors>
    static auto
    broadcast(Arg&& arg, const Factors&...factors)
    {
      static_assert(sizeof...(Factors) == sizeof...(IndexDescriptors));
      decltype(auto) ret = NestedInterface::broadcast(std::forward<Arg>(arg).nested_matrix(), factors...);
      using Ret = decltype(ret);
      return broadcast_impl(std::forward<Ret>(ret), std::make_index_sequence<sizeof...(Factors)>{}, factors...);
    }


    template<typename...Ds, typename Operation, typename...Args>
    static auto n_ary_operation(const std::tuple<Ds...>& d_tup, Operation&& op, Args&&...args)
    {
      return NestedInterface::n_ary_operation(d_tup, std::forward<Operation>(op), std::forward<Args>(args)...);
    }


    template<std::size_t...indices, typename BinaryFunction, typename Arg>
    static constexpr decltype(auto) reduce(BinaryFunction&& b, Arg&& arg)
    {
      return NestedInterface::template reduce<indices...>(std::forward<BinaryFunction>(b), std::forward<Arg>(arg).nested_matrix());
    }

    template<typename Arg, typename C>
    constexpr decltype(auto) to_euclidean(Arg&& arg, const C& c)
    {
      return to_euclidean(std::forward<Arg>(arg).nested_matrix(), c);
    }

    template<typename Arg, typename C>
    constexpr decltype(auto) from_euclidean(Arg&& arg, const C& c)
    {
      return from_euclidean(std::forward<Arg>(arg).nested_matrix(), c);
    }

    template<typename Arg, typename C>
    constexpr decltype(auto) wrap_angles(Arg&& arg, const C& c)
    {
      return wrap_angles(std::forward<Arg>(arg).nested_matrix(), c);
    }

    template<typename Arg>
    static constexpr auto conjugate(Arg&& arg) { return OpenKalman::conjugate(std::forward<Arg>(arg).nested_matrix()); }

    template<typename Arg>
    static constexpr auto transpose(Arg&& arg) { return OpenKalman::transpose(std::forward<Arg>(arg).nested_matrix()); }

    template<typename Arg>
    static constexpr auto adjoint(Arg&& arg) { return OpenKalman::adjoint(std::forward<Arg>(arg).nested_matrix()); }

    template<typename Arg>
    static constexpr auto determinant(Arg&& arg) { return OpenKalman::determinant(std::forward<Arg>(arg).nested_matrix()); }

    template<typename A, typename B>
    static constexpr auto sum(A&& a, B&& b) { return OpenKalman::sum(std::forward<A>(a).nested_matrix(), std::forward<B>(b)); }

    template<typename A, typename B>
    static constexpr auto contract(A&& a, B&& b) { return OpenKalman::contract(std::forward<A>(a).nested_matrix(), std::forward<B>(b)); }

    template<typename A, typename B>
    static constexpr auto contract_in_place(A&& a, B&& b) { return OpenKalman::contract_in_place(std::forward<A>(a).nested_matrix(), std::forward<B>(b)); }

    template<TriangleType t, typename Arg>
    static constexpr auto cholesky_factor(Arg&& arg) { return OpenKalman::Cholesky_factor<t>(std::forward<Arg>(arg).nested_matrix()); }

    template<typename A, typename U, typename Alpha>
    static constexpr auto rank_update_self_adjoint(A&& a, U&& u, Alpha alpha) { return OpenKalman::rank_update_self_adjoint(std::forward<A>(a).nested_matrix(), std::forward<U>(u), alpha); }

    template<TriangleType t, typename A, typename U, typename Alpha>
    static constexpr auto rank_update_triangular(A&& a, U&& u, Alpha alpha) { return OpenKalman::rank_update_triangular<t>(std::forward<A>(a).nested_matrix(), std::forward<U>(u), alpha); }

    template<bool must_be_unique = false, bool must_be_exact = false, typename A, typename B>
    static constexpr auto solve(A&& a, B&& b) { return OpenKalman::solve<must_be_unique, must_be_exact>(std::forward<A>(a).nested_matrix(), std::forward<B>(b)); }

    template<typename Arg>
    static constexpr auto LQ_decomposition(Arg&& arg) { return OpenKalman::LQ_decomposition(std::forward<Arg>(arg).nested_matrix()); }

    template<typename Arg>
    static constexpr auto QR_decomposition(Arg&& arg) { return OpenKalman::QR_decomposition(std::forward<Arg>(arg).nested_matrix()); }
  };

} // namespace interface



#endif //OPENKALMAN_FIXEDSIZEADAPTER_HPP
